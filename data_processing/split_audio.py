from pydub import AudioSegment
import sys, os, math

def split_mp3(file_path: str, n_parts: int, output_dir: str | None = None):
    # Завантажуємо аудіо
    audio = AudioSegment.from_file(file_path, format="mp3")
    duration_ms = len(audio)
    print(f"Загальна тривалість: {duration_ms / 1000:.2f} сек ({duration_ms / 60000:.2f} хв)")

    # Обчислюємо довжину одного шматка
    part_length = math.ceil(duration_ms / n_parts)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = output_dir or os.path.dirname(file_path) or "."

    os.makedirs(output_dir, exist_ok=True)

    # Розбиваємо й експортуємо частини
    for i in range(n_parts):
        start = i * part_length
        end = min((i + 1) * part_length, duration_ms)
        chunk = audio[start:end]
        out_path = os.path.join(output_dir, f"{base_name}_part{i+1}.mp3")
        chunk.export(out_path, format="mp3", bitrate="192k")
        print(f"→ {out_path} ({(end - start) / 1000:.2f} сек)")

    print("✅ Готово.")


split_mp3("output.mp3", 40, "audios")
