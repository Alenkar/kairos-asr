import kairos_asr

file_path = "../test_data/record.wav"

asr = kairos_asr.KairosASR()

####################################################################################

print("\n1. Вывод Отдельных Слов С Прогрессом")
result_arr = []
for item, progress in asr.transcribe_iterative(
        wav_file=file_path,
        return_sentences=False,
        with_progress=True):
    result_arr.append(item)
    print(f"Слово: {item.text} ({item.start} - {item.end}) |"
          f"Прогресс: {progress.percent}% (сегмент {progress.segment}/{progress.total_segments}) "
          f"Оставшееся время: {progress.time_remaining} сек")
# Итоговый массив: list[Word]
print(f"Общее количество слов: {len(result_arr)}")

# Итоговый массив: list[Sentence]
print(f"Общее количество предложений: {len(result_arr)}")

####################################################################################

print("\n2. Вывод Предложений С Прогрессом")
result_arr = []
for item, progress in asr.transcribe_iterative(
        wav_file=file_path,
        return_sentences=True,
        with_progress=True):
    result_arr.append(item)
    print(f"Предложение: {item.text} ({item.start} - {item.end}) | "
          f"Прогресс: {progress.percent}% (сегмент {progress.segment}/{progress.total_segments}) "
          f"Оставшееся время: {progress.time_remaining} сек")
# Итоговый массив: list[Sentence]
print(f"Общее количество предложений: {len(result_arr)}")
