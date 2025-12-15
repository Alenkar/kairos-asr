import kairos_asr

file_path = "../test_data/record.wav"

asr = kairos_asr.KairosASR()

####################################################################################

print("\n1. Вывод и обработка аудио целиком.")
result = asr.transcribe(wav_file=file_path)
print(result.full_text)
print(result.words)
for seq in result.sentences:
    print(seq)
print()

####################################################################################

print("\n2. Вывод Отдельных Слов Без Прогресса")
result_arr = []
for item in asr.transcribe_iterative(
        wav_file=file_path,
        return_sentences=False):
    result_arr.append(item)
    print(f"Слово: {item.text} ({item.start} - {item.end})")

# Итоговый массив: list[Word]
print(f"Общее количество слов: {len(result_arr)}")

####################################################################################

print("\n3. Вывод Предложений Без Прогресса")
result_arr = []
for item in asr.transcribe_iterative(
        wav_file=file_path,
        return_sentences=True):
    result_arr.append(item)
    print(f"Предложение: {item.text} ({item.start} - {item.end})")

# Итоговый массив: list[Sentence]
print(f"Общее количество предложений: {len(result_arr)}")
