import json
import base64
import logging
import os
import re
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import requests
import yaml
import yt_dlp
from dataclasses_json import dataclass_json

BATCH_DELIMITER = "\n<<<SRT_DELIM>>>\n"
TRANSLATION_BATCH_SIZE = 2000
TRANSLATION_CACHE_MAXSIZE = 2048
GROQ_TRANSLATION_BATCH_SEGMENTS = 15
GROQ_BATCH_DELAY = 2.0
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})
HTTP_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
HTTP_SESSION.mount("https://", HTTP_ADAPTER)
HTTP_SESSION.mount("http://", HTTP_ADAPTER)
TRANSLATION_CACHE = OrderedDict()

def _get_cached_translation(text, source_lang, target_lang):
    return TRANSLATION_CACHE.get((text, source_lang, target_lang))


def _cache_translation(text, source_lang, target_lang, translated_text):
    key = (text, source_lang, target_lang)
    TRANSLATION_CACHE[key] = translated_text
    TRANSLATION_CACHE.move_to_end(key)
    if len(TRANSLATION_CACHE) > TRANSLATION_CACHE_MAXSIZE:
        TRANSLATION_CACHE.popitem(last=False)

# --- Custom Exception ---


class SubtitleError(Exception):
    """Custom exception for subtitle generation errors."""
    pass


class DirectoryWatcher(threading.Thread):
    def __init__(self, directory, callback):
        super().__init__()
        self.directory = directory
        self.callback = callback
        self.running = True

    def run(self):
        logging.info(f"Starting directory watcher for {self.directory}")
        initial_files = set(os.listdir(self.directory))
        while self.running:
            current_files = set(os.listdir(self.directory))
            new_files = current_files - initial_files
            if new_files:
                for new_file in new_files:
                    full_path = os.path.join(self.directory, new_file)
                    if os.path.isfile(full_path):
                        logging.info(f"New file detected: {new_file}")
                        self.callback(full_path)
            initial_files = current_files

    def stop(self):
        self.running = False
        logging.info("Stopping directory watcher")


def send_subtitles_payload(subtitle_files):
    """Send multiple subtitle files in one request to the asbplayer HTTP endpoint."""
    http_url = "http://127.0.0.1:8766/asbplayer/load-subtitles"
    if not subtitle_files:
        logging.error("No subtitle files provided for sending.")
        return False
    try:
        response = HTTP_SESSION.post(http_url, json={"files": subtitle_files}, timeout=30)
        if response.status_code == 200:
            logging.info(
                f"Successfully sent {len(subtitle_files)} subtitle files to {http_url}")
            logging.debug(f"requests response: {response.text}")
            return True
        logging.error(
            f"Failed to send subtitles to {http_url}. requests returned code: {response.status_code}")
        logging.error(f"requests response text: {response.text}")
        return False
    except Exception as e:
        logging.error(
            f"An error occurred while sending subtitles via HTTP: {e}")
        return False


def send_subtitles_http(srt_file_path):
    """Send a single subtitle file through the existing HTTP endpoint."""
    if not os.path.exists(srt_file_path):
        logging.error(f"SRT file does not exist: {srt_file_path}")
        return False
    try:
        with open(srt_file_path, 'rb') as f:
            srt_content_bytes = f.read()
        base64_srt = base64.b64encode(srt_content_bytes).decode('utf-8')
        filename = os.path.basename(srt_file_path)
        return send_subtitles_payload([{"name": filename, "base64": base64_srt}])
    except FileNotFoundError as e:
        logging.error(f"SRT file not found: {srt_file_path}")
        return False
    except Exception as e:
        logging.error(f"An error occurred while sending subtitles via HTTP: {e}")
        return False


def _parse_srt_blocks(srt_content):
    blocks = []
    raw_blocks = [block.strip() for block in srt_content.split("\n\n") if block.strip()]
    for raw_block in raw_blocks:
        lines = [line for line in raw_block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        if not lines[0].strip().isdigit():
            continue
        timing_line = lines[1].strip()
        if "-->" not in timing_line:
            continue
        text = " ".join(line.strip() for line in lines[2:] if line.strip())
        blocks.append({
            "index": lines[0].strip(),
            "timing": timing_line,
            "text": text,
        })
    return blocks


def _build_srt_from_blocks(blocks):
    output_lines = []
    for block in blocks:
        output_lines.append(block["index"])
        output_lines.append(block["timing"])
        output_lines.append(block["text"])
        output_lines.append("")
    return "\n".join(output_lines).strip() + "\n"


def _translate_text(text, source_lang, target_lang):
    """Translate a single text block using Google Translate API."""
    if not text or not text.strip():
        return text

    cached_translation = _get_cached_translation(text, source_lang, target_lang)
    if cached_translation is not None:
        return cached_translation

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        response = HTTP_SESSION.get(url, params=params, timeout=30)
        response.raise_for_status()
        translation_data = response.json()
        # Response structure: [[[translated_text, original_text, ...], ...], ...]
        if translation_data and isinstance(translation_data[0], list) and translation_data[0]:
            translated_segments = [segment[0] for segment in translation_data[0] if segment and segment[0]]
            translated_text = "".join(translated_segments).strip()
            _cache_translation(text, source_lang, target_lang, translated_text)
            return translated_text
        return text
    except Exception as e:
        logging.warning(f"Failed to translate text: {e}")
        return text


def _build_translation_batches(texts, max_chars=TRANSLATION_BATCH_SIZE):
    batches = []
    current_batch = []
    current_length = 0

    for text in texts:
        text_length = len(text)
        delimiter_length = len(BATCH_DELIMITER) if current_batch else 0
        if current_batch and current_length + delimiter_length + text_length > max_chars:
            batches.append(current_batch)
            current_batch = []
            current_length = 0

        if current_batch:
            current_length += len(BATCH_DELIMITER)
        current_batch.append(text)
        current_length += text_length

    if current_batch:
        batches.append(current_batch)
    return batches


def _translate_text_batch(texts, source_lang, target_lang):
    """Translate a batch of text blocks using Google Translate API."""
    if not texts:
        return []

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": BATCH_DELIMITER.join(texts),
        }
        response = HTTP_SESSION.get(url, params=params, timeout=30)
        response.raise_for_status()
        translation_data = response.json()
        if translation_data and isinstance(translation_data[0], list) and translation_data[0]:
            translated_combined = "".join(
                segment[0] for segment in translation_data[0] if segment and segment[0]
            ).strip()
            translated_parts = translated_combined.split(BATCH_DELIMITER)
            if len(translated_parts) == len(texts):
                clean_parts = [part.strip() for part in translated_parts]
                for original, translated_text in zip(texts, clean_parts):
                    _cache_translation(original, source_lang, target_lang, translated_text)
                return clean_parts

        raise ValueError("Batch translation response did not split correctly")
    except Exception as e:
        logging.warning(f"Batch translation failed: {e}. Falling back to individual requests.")
        return [_translate_text(text, source_lang, target_lang) for text in texts]


def _translate_text_groq(text, source_lang, target_lang, groq_client, model, temperature, system_prompt):
    if not text or not text.strip():
        return text
    cached = _get_cached_translation(text, source_lang, target_lang)
    if cached is not None:
        return cached
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
        )
        translated = response.choices[0].message.content.strip()
        _cache_translation(text, source_lang, target_lang, translated)
        return translated
    except Exception as e:
        logging.warning(f"Groq translation failed: {e}")
        return text


GROQ_BATCH_MAX_RETRIES = 3
_GROQ_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.+)$", re.MULTILINE)


def _translate_text_batch_groq(texts, source_lang, target_lang, groq_client, model, temperature, system_prompt):
    if not texts:
        return []
    numbered_input = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(texts))
    for attempt in range(GROQ_BATCH_MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": numbered_input},
                ],
                temperature=temperature,
            )
            output = response.choices[0].message.content.strip()
            parsed = {int(m.group(1)): m.group(2).strip() for m in _GROQ_LINE_RE.finditer(output)}
            results = [parsed.get(i + 1, texts[i]) for i in range(len(texts))]
            missing = [i + 1 for i in range(len(texts)) if i + 1 not in parsed]
            if missing:
                logging.warning(f"Groq batch attempt {attempt + 1}: missing segment numbers {missing}, retrying...")
                continue
            for original, translated_text in zip(texts, results):
                _cache_translation(original, source_lang, target_lang, translated_text)
            return results
        except Exception as e:
            if attempt < GROQ_BATCH_MAX_RETRIES - 1:
                logging.warning(f"Groq batch translation attempt {attempt + 1} failed: {e}. Retrying...")
            else:
                logging.warning(f"Groq batch translation failed after {GROQ_BATCH_MAX_RETRIES} attempts: {e}. Keeping originals.")
    return texts


def _build_groq_translation_prompt(source_lang, target_lang, video_title="", channel_name=""):
    if video_title and channel_name:
        context = f' You are translating the YouTube video "{video_title}" by user "{channel_name}".'
    elif video_title:
        context = f' You are translating the YouTube video "{video_title}".'
    else:
        context = ""
    return (
        f"You are a professional subtitle translator.{context}"
        f" The transcription you are translating was automatically generated by openai-whisper and may contain errors."
        f" Translate each numbered subtitle segment from {source_lang} to {target_lang}."
        f" Output each translation on its own line in the exact format: [N] translation"
        f" where N is the segment number. Do not add, skip, or merge segments."
    )


def translate_srt_content(srt_content, source_lang="ja", target_lang="en",
                          translation_service="google", groq_client=None, groq_model=None,
                          temperature=0.0, video_title="", channel_name="",
                          groq_batch_segments=GROQ_TRANSLATION_BATCH_SEGMENTS,
                          groq_batch_delay=GROQ_BATCH_DELAY):
    """Translate all subtitle segments in SRT content."""
    blocks = _parse_srt_blocks(srt_content)
    if not blocks:
        raise SubtitleError("No valid SRT segments found for translation.")

    use_groq = translation_service == "groq" and groq_client is not None
    groq_system_prompt = _build_groq_translation_prompt(source_lang, target_lang, video_title, channel_name) if use_groq else None

    untranslated_texts = []
    untranslated_indices = []
    translated_blocks = [None] * len(blocks)

    for i, block in enumerate(blocks):
        text = block["text"]
        if not text or not text.strip():
            translated_blocks[i] = {
                "index": block["index"],
                "timing": block["timing"],
                "text": text,
            }
            continue

        cached_translation = _get_cached_translation(text, source_lang, target_lang)
        if cached_translation is not None:
            translated_blocks[i] = {
                "index": block["index"],
                "timing": block["timing"],
                "text": cached_translation,
            }
            continue

        untranslated_indices.append(i)
        untranslated_texts.append(text)

    if untranslated_texts:
        translated_results = []
        if use_groq:
            groq_batches = [
                untranslated_texts[i:i + groq_batch_segments]
                for i in range(0, len(untranslated_texts), groq_batch_segments)
            ]
            for batch_index, batch in enumerate(groq_batches, start=1):
                batch_translations = _translate_text_batch_groq(
                    batch, source_lang, target_lang, groq_client, groq_model, temperature, groq_system_prompt
                )
                translated_results.extend(batch_translations)
                logging.info(
                    f"Translated batch {batch_index}/{len(groq_batches)} "
                    f"({len(translated_results)}/{len(untranslated_texts)} uncached segments)"
                )
                if batch_index < len(groq_batches):
                    time.sleep(groq_batch_delay)
        else:
            batches = _build_translation_batches(untranslated_texts)
            for batch_index, batch in enumerate(batches, start=1):
                batch_translations = _translate_text_batch(batch, source_lang, target_lang)
                translated_results.extend(batch_translations)
                logging.info(
                    f"Translated batch {batch_index}/{len(batches)} "
                    f"({len(translated_results)}/{len(untranslated_texts)} uncached segments)"
                )

        for idx, translated_text in zip(untranslated_indices, translated_results):
            translated_blocks[idx] = {
                "index": blocks[idx]["index"],
                "timing": blocks[idx]["timing"],
                "text": translated_text,
            }

    translated_blocks = [block if block is not None else {
        "index": blocks[i]["index"],
        "timing": blocks[i]["timing"],
        "text": blocks[i]["text"],
    } for i, block in enumerate(translated_blocks)]

    return _build_srt_from_blocks(translated_blocks)


def translate_srt_file(input_srt_path, output_srt_path, source_lang="ja", target_lang="en",
                       translation_service="google", groq_client=None, groq_model=None,
                       temperature=0.0, video_title="", channel_name="",
                       groq_batch_segments=GROQ_TRANSLATION_BATCH_SEGMENTS,
                       groq_batch_delay=GROQ_BATCH_DELAY):
    if not os.path.exists(input_srt_path):
        raise SubtitleError(f"SRT file does not exist: {input_srt_path}")
    if video_title and channel_name:
        print(f"Translating \"{video_title}\" by {channel_name}")
    elif video_title:
        print(f"Translating \"{video_title}\"")
    try:
        with open(input_srt_path, "r", encoding="utf-8") as f:
            original_content = f.read()
        translated_content = translate_srt_content(
            original_content, source_lang, target_lang,
            translation_service=translation_service, groq_client=groq_client,
            groq_model=groq_model, temperature=temperature,
            video_title=video_title, channel_name=channel_name,
            groq_batch_segments=groq_batch_segments,
            groq_batch_delay=groq_batch_delay,
        )
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(translated_content)
        return output_srt_path
    except Exception as e:
        raise SubtitleError(f"Failed to translate SRT file: {e}") from e


@dataclass_json
@dataclass
class Config:
    process_locally: bool = False
    whisper_model: str = "turbo"
    RUN_ASB_WEBSOCKET_SERVER: bool = True
    GROQ_API_KEY: str = ""
    model: str = "whisper-large-v3-turbo"
    output_dir: str = "output"
    language: str = "ja"
    skip_language_check: bool = False
    enable_translation: bool = True
    translation_target_language: str = "en"
    translation_service: str = "google"
    whisper_translation_model: str = "whisper-large-v3"
    groq_translation_model: str = "llama-3.3-70b-versatile"
    translation_temperature: float = 0.0
    groq_translation_batch_segments: int = 15
    groq_translation_batch_delay: float = 2.0
    # path_to_watch: str = "./watch"
    cookies: str = ""
    monitor_clipboard: bool = True
    download_lower_audio_quality: bool = False
    downsample_audio: bool = True

    def __init__(self, process_locally=False, GROQ_API_KEY="", whisper_model="turbo", RUN_ASB_WEBSOCKET_SERVER=True, model="whisper-large-v3-turbo", output_dir="output", language="ja", skip_language_check=False, enable_translation=True, translation_target_language="en", translation_service="google", whisper_translation_model="whisper-large-v3", groq_translation_model="llama-3.3-70b-versatile", translation_temperature=0.0, groq_translation_batch_segments=15, groq_translation_batch_delay=2.0, path_to_watch="./watch", cookies="", monitor_clipboard=True, download_lower_audio_quality=False, downsample_audio=True, *args, **kwargs):
        self.process_locally = process_locally
        self.GROQ_API_KEY = GROQ_API_KEY
        self.whisper_model = whisper_model
        self.RUN_ASB_WEBSOCKET_SERVER = RUN_ASB_WEBSOCKET_SERVER
        self.model = model
        self.output_dir = output_dir
        self.language = language
        self.skip_language_check = skip_language_check
        self.enable_translation = enable_translation
        self.translation_target_language = translation_target_language
        self.translation_service = translation_service
        self.whisper_translation_model = whisper_translation_model
        self.groq_translation_model = groq_translation_model
        self.translation_temperature = translation_temperature
        self.groq_translation_batch_segments = groq_translation_batch_segments
        self.groq_translation_batch_delay = groq_translation_batch_delay
        # self.path_to_watch = path_to_watch
        self.cookies = cookies
        self.monitor_clipboard = monitor_clipboard
        self.download_lower_audio_quality = download_lower_audio_quality
        self.downsample_audio = downsample_audio


def parse_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = Config(**yaml.safe_load(file))
    except FileNotFoundError:
        config = Config()
        with open(file_path, 'w') as file:
            yaml.safe_dump(config.to_dict(), file)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    return config


# --- YouTube Functions ---

def download_audio(youtube_url, output_dir="."):
    """Downloads audio from YouTube URL, returns final audio file path."""
    logging.info(f"Attempting to download audio from: {youtube_url}")
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'verbose': False, 'skip_download': True, 'noplaylist': True, 'remote_components': ['ejs:github']}) as ydl:
            info_dict_pre = ydl.extract_info(youtube_url, download=False)
            video_id = info_dict_pre.get('id', 'youtube_audio')
            base_filename = os.path.join(output_dir, video_id)
            logging.info(f"Video ID detected: {video_id}")
    except Exception as e:
        logging.warning(
            f"Could not pre-extract video ID, using default filename: {e}")
        base_filename = os.path.join(output_dir, "youtube_audio")

    ydl_opts = {
        'quiet': False,
        'verbose': False,
        'format': '139/bestaudio/best' if config.download_lower_audio_quality else 'bestaudio/best',
        'outtmpl': f'{base_filename}.%(ext)s',
        'keepvideo': False,
        'noplaylist': True,
        'remote_components': ['ejs:github'],
    }

    if config.cookies:
        ydl_opts['cookiesfrombrowser'] = (config.cookies,)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info("Starting download...")
            info_dict = ydl.extract_info(youtube_url, download=True)
            final_audio_path = ydl.prepare_filename(info_dict)
            if os.path.exists(final_audio_path):
                logging.info(f"Audio download successful: {final_audio_path}")
                video_title = info_dict.get("title", "")
                channel_name = info_dict.get("uploader") or info_dict.get("channel", "")
                return final_audio_path, video_title, channel_name
            else:
                raise SubtitleError(
                    f"Expected audio file not found after download: {final_audio_path}")

    except yt_dlp.utils.DownloadError as e:
        logging.error(f"yt-dlp download error: {e}")
        return None, "", ""
    except Exception as e:
        logging.error(
            f"Error during audio download/extraction: {e}", exc_info=True)
        return None, "", ""


def is_youtube_url(url):
    """Checks if the given URL is a valid YouTube URL."""
    if not url or not isinstance(url, str):
        return False
    youtube_regex = re.compile(
        r'(?:https?:\/\/)?(?:www\.)?'
        r'(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)'
        r'([a-zA-Z0-9_-]{11})'
        r'(?:\S*)?'
    )
    return bool(youtube_regex.match(url))


def timed_input(prompt, timeout=5):
    user_input = [None]

    def get_input():
        user_input[0] = input(prompt)

    input_thread = threading.Thread(target=get_input)
    input_thread.start()
    input_thread.join(timeout)

    if input_thread.is_alive():
        logging.info("Input timed out.")
        return None
    return user_input[0]


def is_language_desired(url, desired='ja'):
    """
    Checks if the YouTube video is in desired language.
    """
    if config.skip_language_check:
        logging.info("Skipping language check due to config.skip_language_check=true.")
        return True

    logging.info("Checking video language...")
    yt_dlp_ops = {'quiet': True, 'verbose': False}
    if config.cookies:
        yt_dlp_ops['cookiesfrombrowser'] = (config.cookies,)
    try:
        with yt_dlp.YoutubeDL(yt_dlp_ops) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            # Extract language metadata if available
            language = info_dict.get('language', None)
            if language == desired:  # 'ja' is the language code for Japanese
                return True
            else:
                print(
                    f"Video language {language}, does not match desired language '{desired}'.")
                override = timed_input(
                    "Override language check? Will timeout in 15 seconds. (y/n): ", timeout=15)
                if override and override.strip().lower() in ['y', 'yes']:
                    logging.info("Language check overridden by user.")
                    return True
                else:
                    logging.info("Skipping video due to language mismatch.")
                return False
    except Exception as e:
        logging.error(f"Error checking video language: {e}", exc_info=True)
        override = timed_input(
            "Language check failed due to an error. Override and continue? Will timeout in 15 seconds. (y/n): ",
            timeout=15
        )
        if override and override.strip().lower() in ['y', 'yes']:
            logging.info("Language check error overridden by user.")
            return True
        logging.info("Skipping video due to language check error.")
    return False


def is_file_path(path):
    """Checks if the given path is a valid file path."""
    path = path.replace('"', "")
    if not path or not isinstance(path, str):
        return False
    return os.path.isfile(path) and os.path.exists(path)


def extract_audio_from_local_video(path):
    """Extracts audio from a local video file."""
    if not is_file_path(path):
        logging.error(f"Invalid file path: {path}")
        return None

    output_audio_path = f"{os.path.splitext(path)[0]}.mp3"
    try:
        subprocess.run(["ffmpeg", "-i", path, "-q:a", "0",
                       "-map", "a", output_audio_path], check=True)
        logging.info(f"Audio extracted successfully: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e}")
        return None


class StableTSProcessor:
    """
    Processor to run stable-ts on a local audio file and return segment/word timestamps similar to groq output.
    """

    def __init__(self, model="turbo", extra_args=None):
        self.model = model
        self.extra_args = extra_args or []

        try:
            import torch
        except ImportError as exc:
            raise SubtitleError(
                "Local transcription requires optional dependency 'torch'. "
                "Install `asb-auto-subgen[local]` or set `process_locally: false` in config.yaml."
            ) from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            import stable_whisper
        except ImportError as exc:
            raise SubtitleError(
                "Local transcription requires `stable-ts`. "
                "Install `asb-auto-subgen[local]` or set `process_locally: false` in config.yaml."
            ) from exc
        try:
            self.model = stable_whisper.load_model(self.model, device=self.device)
        except Exception as e:
            logging.error(f"Failed to load stable-ts model: {e}")
            raise SubtitleError(f"Failed to load stable-ts model: {e}")

    def get_audio_segments(self, audio_path, language="ja", word_timestamps=False, vad=True, min_silence_duration_ms=250):
        """
        Run stable-ts (via stable_whisper) on the given audio file and return parsed segments/words.
        Returns a dict with 'segments' and 'words' keys, similar to groq output.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Transcribe
        try:
            result = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                vad=vad,
                temperature=0.0,
                # Add any extra args if needed
            )
        except Exception as e:
            logging.error(f"stable-ts transcription failed: {e}")
            raise SubtitleError(f"stable-ts transcription failed: {e}")

        # Convert to groq-like format
        segments = []
        words = []
        for i, seg in enumerate(result.segments):
            segments.append({
                "id": i,
                "start": float(seg.start) if hasattr(seg, 'start') else 0.0,
                "end": float(seg.end) if hasattr(seg, 'end') else 0.0,
                "text": getattr(seg, 'text', "")
            })
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    words.append({
                        "id": len(words),
                        "start": float(getattr(w, 'start', 0.0)),
                        "end": float(getattr(w, 'end', 0.0)),
                        "word": getattr(w, 'word', getattr(w, 'text', ""))
                    })

        return {"segments": segments, "words": words}


config = parse_config('config.yaml')
