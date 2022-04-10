import graph_utils.chat as pla
import os
import fasttext

def file_to_string(path, remove_stopwords=False, remove_nonalpha=True):
    chat = pla.read_chat(path)
    sents = chat.sents(participant='PAR')

    s = ''
    for sent in sents:
        if remove_stopwords:
            sent = [word for word in sent if word not in stopwords.words('english')]
        if remove_nonalpha:
            sent = list([''.join(ch for ch in word if ch.isalpha()) for word in sent])
        sent = list(filter(lambda x: len(x) > 0, sent))
        s += ' '.join(sent)
        s += '\n'

    return s


def main():
    s = ''
    print('Unreadable chat files:')
    for group in ['Controls', 'Dementia']:
        for f in os.listdir(f'data/DementiaBank/{group}/'):
            try:
                s += file_to_string(os.path.join(f'data/DementiaBank/{group}', f))
            except StopIteration:
                print(f)
            s += '\n'
    for group in ['cc', 'cd']:
        path = f'data/ADReSS-IS2020-data/train/transcription/{group}/'
        for f in os.listdir(path):
            try:
                s += file_to_string(os.path.join(path, f))
            except StopIteration:
                print(f)
            s += '\n'
    s += s
    s = s.encode(encoding='UTF-8')
    with open('results/transcript_text.txt', 'wb') as f:
        f.write(s)
    model = fasttext.train_unsupervised('results/transcript_text.txt', minn=2, maxn=6, dim=300)
    model.save_model('results/DementiaBank.bin')


if __name__ == '__main__':
    main()
