import io
import os
import string
import re
import pickle

data_path = 'data'


def get_data(path):
    final_data = []
    for file in os.listdir(path):
        try:
            f = io.open(os.path.join(path, file), mode="r",encoding="utf8", errors='ignore')
            text = f.read()
            printable = set(string.printable)
            cleaned_text = ''.join(filter(lambda x: x in printable, text))
            cleaned_text = cleaned_text.replace('\x0b', ' ').replace('\x0c', ' ').replace('\n', ' ')
            corrected_gt = re.findall('(?<=CorrectedAsrTranscript \*)(.*?)\*', cleaned_text)
            transcribed = re.findall('(?<=[\s|\t|z]AsrTranscript \*)(.*?)\*', cleaned_text)
            data = []
            for original, correct in zip(transcribed, corrected_gt):
                data.append((original.replace('zLlmCorrectedTranscript', '').strip(),
                             correct.replace('zFinalTranscript', '').replace('z FinalTranscript', '').replace('z\tFinalTranscript', '').strip()))
            final_data.extend(data)
            with open('export/data.pkl', 'wb') as file_out:
                pickle.dump(final_data, file_out)
        except Exception:
            print(f'No able to read {file}. Skipped.')


if __name__ == '__main__':
    get_data(data_path)
