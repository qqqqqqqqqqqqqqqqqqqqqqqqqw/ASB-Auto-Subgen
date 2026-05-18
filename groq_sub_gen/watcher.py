import base64
import os
import subprocess
import json
import re
import logging
import time
import requests
from groq import Groq # Assuming groq library is installed and used

try:
    import yt_dlp
    import pyperclip
    import groq
except ImportError as e:
    print(f"Error: Missing dependency - {e.name}")
    print("Please install required libraries: pip install yt-dlp pyperclip groq")
    exit(1)

from groq_sub_gen.shared import *

# --- Configuration ---
# Recommended: Load API key from environment variable
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Alternative (less secure):

# Directory to save downloaded audio and generated SRT files
OUTPUT_DIR = config.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants (from SubtitleProcessor) ---

LANGUAGE_CODES = {
    "English": "en", "Chinese": "zh", "German": "de", "Spanish": "es", "Russian": "ru",
    "Korean": "ko", "French": "fr", "Japanese": "ja", "Portuguese": "pt", "Turkish": "tr",
    "Polish": "pl", "Catalan": "ca", "Dutch": "nl", "Arabic": "ar", "Swedish": "sv",
    "Italian": "it", "Indonesian": "id", "Hindi": "hi", "Finnish": "fi", "Vietnamese": "vi",
    "Hebrew": "he", "Ukrainian": "uk", "Greek": "el", "Malay": "ms", "Czech": "cs",
    "Romanian": "ro", "Danish": "da", "Hungarian": "hu", "Tamil": "ta", "Norwegian": "no",
    "Thai": "th", "Urdu": "ur", "Croatian": "hr", "Bulgarian": "bg", "Lithuanian": "lt",
    "Latin": "la", "Māori": "mi", "Malayalam": "ml", "Welsh": "cy", "Slovak": "sk",
    "Telugu": "te", "Persian": "fa", "Latvian": "lv", "Bengali": "bn", "Serbian": "sr",
    "Azerbaijani": "az", "Slovenian": "sl", "Kannada": "kn", "Estonian": "et",
    "Macedonian": "mk", "Breton": "br", "Basque": "eu", "Icelandic": "is", "Armenian": "hy",
    "Nepali": "ne", "Mongolian": "mn", "Bosnian": "bs", "Kazakh": "kk", "Albanian": "sq",
    "Swahili": "sw", "Galician": "gl", "Marathi": "mr", "Panjabi": "pa", "Sinhala": "si",
    "Khmer": "km", "Shona": "sn", "Yoruba": "yo", "Somali": "so", "Afrikaans": "af",
    "Occitan": "oc", "Georgian": "ka", "Belarusian": "be", "Tajik": "tg", "Sindhi": "sd",
    "Gujarati": "gu", "Amharic": "am", "Yiddish": "yi", "Lao": "lo", "Uzbek": "uz",
    "Faroese": "fo", "Haitian": "ht", "Pashto": "ps", "Turkmen": "tk", "Norwegian Nynorsk": "nn",
    "Maltese": "mt", "Sanskrit": "sa", "Luxembourgish": "lb", "Burmese": "my", "Tibetan": "bo",
    "Tagalog": "tl", "Malagasy": "mg", "Assamese": "as", "Tatar": "tt", "Hawaiian": "haw",
    "Lingala": "ln", "Hausa": "ha", "Bashkir": "ba", "Javanese": "jw", "Sundanese": "su",
}

ALLOWED_FILE_EXTENSIONS = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
MAX_FILE_SIZE_MB = 10
CHUNK_SIZE_MB = 10

# --- Custom Exception ---
class SubtitleError(Exception):
    """Custom exception for subtitle generation errors."""
    pass

# --- Subtitle Processor Class (Previously Refactored) ---
class SubtitleProcessor:
    def __init__(self, groq_client=None, local_processor=None):
        if not groq_client and not local_processor:
            raise ValueError("At least one of 'groq_client' or 'local_processor' must be provided.")
        self.client = groq_client
        self.local_processor: StableTSProcessor = local_processor
        self._temp_files = []

    def _run_command(self, cmd_list):
        try:
            process = subprocess.run(
                cmd_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            logging.debug(f"Command successful: {' '.join(cmd_list)}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}")
        except subprocess.CalledProcessError as e:
            error_message = (f"Command failed: {' '.join(e.cmd)}\nReturn code: {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}")
            logging.error(error_message)
            raise SubtitleError(f"Error during command execution: {' '.join(cmd_list)}. Check logs.") from e
        except FileNotFoundError as e:
             logging.error(f"Command not found: {cmd_list[0]}. Is ffmpeg installed and in PATH?")
             raise SubtitleError(f"Required command not found: {cmd_list[0]}. Ensure ffmpeg is accessible.") from e

    def _cleanup_temp_files(self):
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    logging.info(f"Cleaned up temporary file: {f}")
            except OSError as e:
                logging.warning(f"Could not remove temporary file {f}: {e}")
        self._temp_files = []

    def _process_audio_chunk_with_retry(self, current_file_path, timestamp_granularities_list, language, auto_detect_language, model, prompt, max_retries=3):
        """Process a single audio chunk with retry logic and exponential backoff."""
        for attempt in range(max_retries):
            try:
                with open(current_file_path, "rb") as file_data:
                    if not self.local_processor:
                        transcription_response = self.client.audio.transcriptions.create(
                            file=(os.path.basename(current_file_path), file_data.read()),
                            model=model, prompt=prompt, response_format="verbose_json",
                            timestamp_granularities=timestamp_granularities_list,
                            language=None if auto_detect_language else language, temperature=0.0,
                        )
                    else: 
                        transcription_response = self.local_processor.get_audio_segments(
                            current_file_path, language=language, word_timestamps=("word" in timestamp_granularities_list),
                        )
                return transcription_response
            except (groq.RateLimitError, groq.APIStatusError) as e:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff: 1s, 2s, 4s, etc., max 60s
                if attempt < max_retries - 1:
                    logging.warning(f"Attempt {attempt + 1}/{max_retries} failed for {os.path.basename(current_file_path)}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All {max_retries} retry attempts failed for {os.path.basename(current_file_path)}.")
                    raise
            except Exception as e:
                logging.error(f"Non-retryable error processing {os.path.basename(current_file_path)}: {e}")
                raise

    def _handle_groq_error(self, e, model_name):
        try:
            error_data = e.args[0]
            error_message = f"Groq API Error ({type(e).__name__}) with model {model_name}"
            if isinstance(error_data, str):
                json_match = re.search(r'(\{.*\})', error_data)
                if json_match:
                    try:
                        json_str = json_match.group(1).replace("'", '"')
                        error_data = json.loads(json_str)
                    except json.JSONDecodeError: error_message += f": Could not parse error details: {error_data}"
                else: error_message += f": {error_data}"
            if isinstance(error_data, dict) and 'error' in error_data and 'message' in error_data['error']:
                 api_msg = error_data['error']['message']
                 api_msg = re.sub(r'org_[a-zA-Z0-9]+', 'org_(censored)', api_msg)
                 error_message += f": {api_msg}"
            elif isinstance(error_data, str) and not json_match: error_message += f": {error_data}"
        except Exception as parse_exc:
            logging.error(f"Failed to parse Groq error details: {parse_exc}")
            error_message = f"Unknown Groq API error occurred: {e}"
        raise SubtitleError(error_message) from e

    def _split_audio(self, input_file_path, chunk_size_mb):
        """Split audio file into chunks using ffmpeg for proper audio boundaries."""
        chunks = []
        base_name, extension = os.path.splitext(input_file_path)
        try:
            # First, get the duration of the audio file
            probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", input_file_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_result.stdout)
            duration = float(probe_data['format']['duration'])
            
            # Calculate chunk duration based on actual file bitrate
            file_size_bytes = os.path.getsize(input_file_path)
            bitrate_bps = (file_size_bytes * 8) / duration
            chunk_duration = (chunk_size_mb * 1024 * 1024 * 8) / bitrate_bps
            
            num_chunks = int(duration // chunk_duration) + 1
            actual_chunk_duration = duration / num_chunks
            
            logging.info(f"Audio duration: {duration:.2f}s, bitrate: {bitrate_bps/1000:.0f}kbps, splitting into {num_chunks} chunks of ~{actual_chunk_duration:.2f}s each")
            
            for i in range(num_chunks):
                start_time = i * actual_chunk_duration
                chunk_name = f"{base_name}_part{i+1:03}{extension}"
                
                # Use ffmpeg to extract segment without re-encoding
                cmd = [
                    "ffmpeg", "-y", "-i", input_file_path,
                    "-ss", str(start_time), "-t", str(actual_chunk_duration),
                    "-c", "copy", chunk_name
                ]
                self._run_command(cmd)
                chunks.append(chunk_name)
                self._temp_files.append(chunk_name)
                
            if not chunks:
                raise SubtitleError(f"No chunks created from {input_file_path}.")
            logging.info(f"Split {input_file_path} into {len(chunks)} chunks using ffmpeg.")
            return chunks
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logging.warning(f"FFmpeg splitting failed: {e}. Falling back to byte-based splitting.")
            # Fallback to original byte-based splitting if ffmpeg fails
            chunk_size = int(chunk_size_mb * 1024 * 1024)
            file_number = 1
            chunks = []
            try:
                with open(input_file_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk: break
                        chunk_name = f"{base_name}_part{file_number:03}{extension}"
                        with open(chunk_name, 'wb') as chunk_file: chunk_file.write(chunk)
                        chunks.append(chunk_name)
                        self._temp_files.append(chunk_name)
                        file_number += 1
                if not chunks: raise SubtitleError(f"No chunks created from {input_file_path}.")
                logging.info(f"Split {input_file_path} into {len(chunks)} chunks (byte-based fallback).")
                return chunks
            except IOError as e: raise SubtitleError(f"Error splitting file {input_file_path}: {e}") from e
        except Exception as e:
            raise SubtitleError(f"Unexpected error during audio splitting: {e}") from e

    def _merge_files(self, chunks, output_file_path):
        if not chunks: return
        list_file_path = "temp_ffmpeg_list.txt"
        try:
            with open(list_file_path, "w", encoding='utf-8') as f:
                for file in chunks:
                    safe_file_path = file.replace("'", "'\\''")
                    f.write(f"file '{safe_file_path}'\n")
            self._temp_files.append(list_file_path)
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-c", "copy", output_file_path]
            self._run_command(cmd)
            logging.info(f"Successfully merged {len(chunks)} chunks into {output_file_path}")
        except Exception as e:
            logging.error(f"Error during file merging: {e}")
            try:
                 if os.path.exists(list_file_path): os.remove(list_file_path)
            except OSError: logging.warning(f"Could not remove temp list file: {list_file_path}")
            raise SubtitleError(f"Failed to merge files into {output_file_path}") from e
        finally:
             if list_file_path not in self._temp_files: self._temp_files.append(list_file_path)

    def _check_and_prepare_file(self, input_file_path, split=False):
        if not input_file_path or not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        try: file_size_mb = os.path.getsize(input_file_path) / (1024 * 1024)
        except OSError as e: raise SubtitleError(f"Could not get size of file {input_file_path}: {e}") from e
        file_extension = os.path.splitext(input_file_path)[1].lower().lstrip('.')
        if file_extension not in ALLOWED_FILE_EXTENSIONS:
            raise ValueError(f"Invalid file type (.{file_extension}). Allowed: {', '.join(ALLOWED_FILE_EXTENSIONS)}")
        if not split:
            return input_file_path, None  # No processing needed if not splitting
        
        if file_size_mb <= MAX_FILE_SIZE_MB:
            logging.info(f"File '{os.path.basename(input_file_path)}' ({file_size_mb:.2f} MB) within size limit.")
            return input_file_path, None
        if not config.downsample_audio:
            logging.warning(f"File ({file_size_mb:.2f} MB) > limit ({MAX_FILE_SIZE_MB} MB). Downsampling disabled — splitting directly.")
            try:
                chunks = self._split_audio(input_file_path, CHUNK_SIZE_MB)
                return chunks, "split"
            except (OSError, SubtitleError) as e:
                raise SubtitleError(f"Error during file preparation: {e}") from e
        logging.warning(f"File ({file_size_mb:.2f} MB) > limit ({MAX_FILE_SIZE_MB} MB). Attempting downsample.")
        output_file_path = os.path.splitext(input_file_path)[0] + "_downsampled.mp3"
        cmd = ["ffmpeg", "-y", "-i", input_file_path, "-ar", "16000", "-ab", "128k", "-ac", "1", "-f", "mp3", output_file_path]
        try:
            self._run_command(cmd)
            self._temp_files.append(output_file_path)
            downsampled_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
            if downsampled_size_mb <= MAX_FILE_SIZE_MB:
                logging.info(f"Downsampled '{os.path.basename(output_file_path)}' size: {downsampled_size_mb:.2f} MB.")
                return output_file_path, "downsampled"
            else:
                logging.warning(f"Still too large ({downsampled_size_mb:.2f} MB). Splitting into {CHUNK_SIZE_MB} MB chunks.")
                chunks = self._split_audio(output_file_path, CHUNK_SIZE_MB)
                return chunks, "split"
        except (OSError, SubtitleError) as e:
             raise SubtitleError(f"Error during file preparation: {e}") from e

    @staticmethod
    def _format_time(seconds_float):
        total_seconds = int(seconds_float)
        milliseconds = int(round((seconds_float - total_seconds) * 1000))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    @staticmethod
    def _json_to_srt(transcription_json_segments):
        srt_lines = []
        for segment in transcription_json_segments:
            start_time = SubtitleProcessor._format_time(segment.get('start', 0.0))
            end_time = SubtitleProcessor._format_time(segment.get('end', 0.0))
            text = segment.get('text', '').strip()
            srt_lines.append(f"{segment.get('id', -1) + 1}\n{start_time} --> {end_time}\n{text}\n")
        return '\n'.join(srt_lines)

    @staticmethod
    def _words_json_to_srt(words_data, starting_id=0):
        srt_lines = []
        previous_end_time = 0.0
        min_duration = 0.050
        for i, word_entry_obj in enumerate(words_data):
            word_entry = word_entry_obj if isinstance(word_entry_obj, dict) else word_entry_obj.__dict__
            start_seconds = word_entry.get('start', 0.0)
            end_seconds = word_entry.get('end', 0.0)
            text = word_entry.get('word', '').strip()
            start_seconds = max(start_seconds, previous_end_time)
            if end_seconds <= start_seconds: end_seconds = start_seconds + min_duration
            start_time_fmt = SubtitleProcessor._format_time(start_seconds)
            end_time_fmt = SubtitleProcessor._format_time(end_seconds)
            srt_id = starting_id + i + 1
            srt_lines.append(f"{srt_id}\n{start_time_fmt} --> {end_time_fmt}\n{text}\n")
            previous_end_time = end_seconds
        return '\n'.join(srt_lines)

    def generate_subtitles(
        self,
        input_file_path: str,
        output_srt_path: str,
        output_video_path: str = None, # Kept for compatibility, but not used in YT flow
        prompt: str = "",
        timestamp_granularities_str: str = "segment",
        language: str = "ja",
        auto_detect_language: bool = False,
        model: str = config.model,
        include_video: bool = False, # Kept for compatibility
        # Font options kept for compatibility but ignored if include_video is False
        font_selection: str = "Default", font_file_path: str = None,
        font_color: str = "#FFFFFF", font_size: int = 24,
        outline_thickness: int = 1, outline_color: str = "#000000",
        chunk_callback=None,
    ):
        self._temp_files = []
        processed_path_or_chunks = None
        base_filename = os.path.splitext(os.path.basename(output_srt_path))[0]

        try:
            if not auto_detect_language and language not in LANGUAGE_CODES.values():
                 raise ValueError(f"Invalid language code '{language}'. Check LANGUAGE_CODES.")
            # Skip font validation if not embedding video
            # ...
        
            processed_path_or_chunks, status = self._check_and_prepare_file(input_file_path, split=self.local_processor is None)
            is_split = (status == "split")

            full_srt_content_list = []
            total_duration_offset = 0.0
            srt_entry_offset = 0

            files_to_process = processed_path_or_chunks if is_split else [processed_path_or_chunks]
            # input_is_video = input_file_path.lower().endswith((".mp4", ".webm", ".mov")) # Less relevant now

            timestamp_granularities_list = [gran.strip() for gran in timestamp_granularities_str.split(',') if gran.strip()]
            if not timestamp_granularities_list or not all(g in ["segment", "word"] for g in timestamp_granularities_list):
                raise ValueError("Invalid timestamp_granularities_str. Use 'segment', 'word', or 'segment,word'.")
            primary_granularity = "word" if "word" in timestamp_granularities_list else "segment"
            logging.info(f"Using primary timestamp granularity: {primary_granularity}")

            for i, current_file_path in enumerate(files_to_process):
                logging.info(f"Processing {'chunk' if is_split else 'file'} {i + 1}/{len(files_to_process)}: {os.path.basename(current_file_path)}")
                chunk_srt_content = ""
                try:
                    # Use retry logic for API calls
                    transcription_response = self._process_audio_chunk_with_retry(
                        current_file_path, timestamp_granularities_list, language, auto_detect_language, model, prompt
                    )
                    
                    # Normalize to a dict for both local and remote.
                    transcription_response = dict(transcription_response)
                    word_data = getattr(transcription_response, 'words', transcription_response.get('words', []))
                    segment_data = getattr(transcription_response, 'segments', transcription_response.get('segments', []))
                    
                    if primary_granularity == "word":
                        if word_data:
                            adjusted_word_data = []
                            last_end_time_in_chunk = 0.0
                            for entry_obj in word_data:
                                entry = entry_obj if isinstance(entry_obj, dict) else entry_obj.__dict__
                                adjusted_entry = entry.copy()
                                start = adjusted_entry.get('start', 0.0)
                                end = adjusted_entry.get('end', 0.0)
                                adjusted_entry['start'] = start + total_duration_offset
                                adjusted_entry['end'] = end + total_duration_offset
                                last_end_time_in_chunk = max(last_end_time_in_chunk, adjusted_entry['end'])
                                adjusted_word_data.append(adjusted_entry)
                            chunk_srt_content = self._words_json_to_srt(adjusted_word_data, srt_entry_offset)
                            total_duration_offset = last_end_time_in_chunk # Update offset based on max end time in this chunk
                            srt_entry_offset += len(word_data)
                        else: logging.warning(f"API returned no word timestamps for {os.path.basename(current_file_path)}.")

                    elif primary_granularity == "segment":
                         if segment_data:
                            adjusted_segment_data = []
                            max_original_id = -1
                            last_end_time_in_chunk = 0.0
                            for entry_obj in segment_data:
                                entry = entry_obj if isinstance(entry_obj, dict) else entry_obj.__dict__
                                adjusted_entry = entry.copy()
                                start = adjusted_entry.get('start', 0.0)
                                end = adjusted_entry.get('end', 0.0)
                                adjusted_entry['start'] = start + total_duration_offset
                                adjusted_entry['end'] = end + total_duration_offset
                                last_end_time_in_chunk = max(last_end_time_in_chunk, adjusted_entry['end'])
                                original_id = entry.get('id', -1)
                                max_original_id = max(max_original_id, original_id)
                                adjusted_entry['id'] = original_id # Keep for offset calc
                                adjusted_segment_data.append(adjusted_entry)
                            # Adjust IDs sequentially for SRT generation
                            for j, entry in enumerate(adjusted_segment_data): entry['id'] = srt_entry_offset + j
                            chunk_srt_content = self._json_to_srt(adjusted_segment_data)
                            total_duration_offset = last_end_time_in_chunk # Update offset
                            srt_entry_offset += (max_original_id + 1)
                         else: logging.warning(f"API returned no segment timestamps for {os.path.basename(current_file_path)}.")

                    if chunk_srt_content:
                        full_srt_content_list.append(chunk_srt_content)
                        if chunk_callback:
                            current_full_srt = "\n".join(full_srt_content_list)
                            subtitle_payloads = [{
                                "name": f"{base_filename}.srt",
                                "base64": base64.b64encode(current_full_srt.encode("utf-8")).decode("utf-8"),
                            }]
                            if config.enable_translation:
                                try:
                                    translated_full_srt = translate_srt_content(
                                        current_full_srt,
                                        source_lang=language,
                                        target_lang=config.translation_target_language,
                                    )
                                    subtitle_payloads.append({
                                        "name": f"{base_filename}.{config.translation_target_language}.srt",
                                        "base64": base64.b64encode(translated_full_srt.encode("utf-8")).decode("utf-8"),
                                    })
                                except Exception as translation_err:
                                    logging.warning(
                                        f"Full SRT translation failed after chunk {i+1}: {translation_err}. Sending original only."
                                    )
                            try:
                                logging.info(f"Sending updated full subtitles after chunk {i+1}/{len(files_to_process)} to asbplayer.")
                                chunk_callback(subtitle_payloads)
                            except Exception as callback_err:
                                logging.warning(
                                    f"Failed to send updated subtitles after chunk {i+1}: {callback_err}"
                                )

                except groq.AuthenticationError as e:
                    self._handle_groq_error(e, model)
                except groq.GroqError as e:
                    logging.error(f"Groq API error processing {os.path.basename(current_file_path)}: {e}", exc_info=True)
                    logging.warning(f"Skipping chunk {i+1} due to error.")
                    continue
                except Exception as e:
                    logging.error(f"Error processing {os.path.basename(current_file_path)}: {e}", exc_info=True)
                    logging.warning(f"Skipping chunk {i+1} due to error.")
                    continue

            if not full_srt_content_list:
                logging.warning("No subtitle content was generated.")
                self._cleanup_temp_files()
                return None, None # Return None for SRT path

            final_srt_content = "\n".join(full_srt_content_list)
            try:
                with open(output_srt_path, "w", encoding="utf-8") as f:
                    f.write(final_srt_content)
                logging.info(f"Successfully generated SRT file: {output_srt_path}")
            except IOError as e:
                raise SubtitleError(f"Failed to write final SRT file to {output_srt_path}: {e}") from e

            # Video embedding part is skipped as include_video=False implicitly for YT audio
            final_video_output = None

            return output_srt_path, final_video_output # Return SRT path and None for video

        except (FileNotFoundError, ValueError, SubtitleError, groq.GroqError) as e:
            logging.error(f"Subtitle generation failed: {e}", exc_info=False)
            raise
        except Exception as e:
            logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
            raise SubtitleError("An unexpected critical error occurred.") from e
        finally:
             self._cleanup_temp_files()

    # _embed_subtitles method is omitted as it's not used in this workflow


async def main(url=None):
    try:
        groq_client = Groq(api_key=config.GROQ_API_KEY, timeout=600)  # Increased timeout to 10 minutes
        if config.process_locally:
            logging.info("Processing subtitles locally.")
            processor = SubtitleProcessor(
                local_processor=StableTSProcessor(model=config.whisper_model)
            )
        else:
            if not config.GROQ_API_KEY:
                logging.error("GROQ_API_KEY not set. Cannot proceed.")
                return
            processor = SubtitleProcessor(groq_client=groq_client)
        logging.info("Groq client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize subtitle processing: {e}")
        return

    if not config.monitor_clipboard:
        if not url:
            logging.error("monitor_clipboard is disabled in config.yaml. Provide a YouTube URL as a CLI argument.")
            return
        if not is_youtube_url(url):
            logging.error(f"Invalid YouTube URL: {url}")
            return
        logging.info(f"Processing URL from argument: {url}")
        audio_file_path = None
        try:
            audio_file_path = download_audio(url, OUTPUT_DIR)
            if audio_file_path and os.path.exists(audio_file_path):
                get_subs(processor, audio_file_path)
            else:
                logging.error("Audio download failed or file not found.")
        finally:
            if audio_file_path and os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                    logging.info(f"Cleaned up downloaded audio file: {audio_file_path}")
                except OSError as e:
                    logging.warning(f"Could not remove audio file {audio_file_path}: {e}")
        return

    previous_clipboard_content = ""
    logging.info("Monitoring clipboard for YouTube links... (Press Ctrl+C to stop)")

    while True:
        try:
            current_clipboard_content = pyperclip.paste()
            if current_clipboard_content != previous_clipboard_content and current_clipboard_content:
                previous_clipboard_content = current_clipboard_content
                if is_youtube_url(current_clipboard_content):
                    logging.info(f"Detected YouTube link: {current_clipboard_content}")
                    if is_language_desired(current_clipboard_content, 'ja'):
                        audio_file_path = None
                        try:
                            audio_file_path = download_audio(current_clipboard_content, OUTPUT_DIR)

                            if audio_file_path and os.path.exists(audio_file_path):
                                logging.info(f"Audio downloaded to: {audio_file_path}")
                                get_subs(processor, audio_file_path)

                            else:
                                logging.error("Audio download failed or file not found.")

                        finally:
                            if audio_file_path and os.path.exists(audio_file_path):
                                try:
                                    os.remove(audio_file_path)
                                    logging.info(f"Cleaned up downloaded audio file: {audio_file_path}")
                                except OSError as e:
                                    logging.warning(f"Could not remove downloaded audio file {audio_file_path}: {e}")
                elif is_file_path(current_clipboard_content):
                    path = current_clipboard_content.strip().replace('"', '').replace("'", "")
                    try:
                        audio = extract_audio_from_local_video(path)
                    except Exception as e:
                        logging.error(f"Error extracting audio from local video: {e}")
                        return
                    
                    if audio and os.path.exists(audio):
                        get_subs(processor, audio)
                        logging.info(f"Audio extracted from local video: {audio}")
                    else:
                        logging.error("Audio extraction failed.")
                        return
            time.sleep(1)

        except pyperclip.PyperclipException as clip_err:
            logging.warning(f"Could not access clipboard: {clip_err}. Retrying...")
            time.sleep(5)
        except KeyboardInterrupt:
            logging.info("Stopping clipboard monitoring.")
            break
        except Exception as loop_err:
            logging.error(f"Error in main loop: {loop_err}", exc_info=True)
            time.sleep(5)

def get_subs(processor, audio_file_path):
        processor: SubtitleProcessor
        base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_srt_path = os.path.join(OUTPUT_DIR, f"{base_filename}.srt")

        def chunk_callback(subtitle_payloads):
            if not subtitle_payloads:
                return False
            logging.info(f"Sending subtitle chunk with {len(subtitle_payloads)} file(s) to asbplayer.")
            success = send_subtitles_payload(subtitle_payloads)
            if not success:
                logging.warning("Subtitle chunk send failed.")
            return success

        try:
            srt_path, _ = processor.generate_subtitles(
                                            input_file_path=audio_file_path,
                                            output_srt_path=output_srt_path,
                                            timestamp_granularities_str="segment",
                                            language='ja',
                                            auto_detect_language=False,
                                            include_video=False,
                                            chunk_callback=chunk_callback,
                                        )
            if srt_path:
                logging.info(f"Subtitles generated successfully: {srt_path}")
                if config.enable_translation:
                    translated_srt_path = os.path.join(OUTPUT_DIR, f"{base_filename}.{config.translation_target_language}.srt")
                    try:
                        translate_srt_file(
                            srt_path,
                            translated_srt_path,
                            source_lang=config.language,
                            target_lang=config.translation_target_language,
                        )
                        logging.info(f"Translated subtitle file written to: {translated_srt_path}")
                    except SubtitleError as translate_err:
                        logging.warning(f"Failed to write translated subtitle file: {translate_err}")
                logging.info("Chunked subtitle delivery completed.")
            else:
                logging.error("Subtitle generation failed (returned None).")

        except (SubtitleError, ValueError, groq.GroqError) as sub_err:
            logging.error(f"Error during subtitle generation: {sub_err}")
        except Exception as gen_err:
            logging.error(f"Unexpected error during subtitle generation: {gen_err}", exc_info=True)

def test_send():
    send_subtitles_http("Hipe5_osY-k.srt")

if __name__ == "__main__":


    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: ffmpeg command not found or failed execution.")
        print("         Ensure ffmpeg is installed and in your system's PATH for audio extraction/conversion.")

    # test_send()
    main()

    # End of the main.py script
