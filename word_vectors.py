import millington.chat as pla
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

    s = file_to_string('ADReSS-IS2020-data/train/transcription/cc/*')
    s += '\n' + file_to_string('ADReSS-IS2020-data/train/transcription/cd/*')
    s += s
    s += s
    s += s
    s = s.encode(encoding='UTF-8')
    with open('word_vectors/transcript_text.txt', 'wb') as f:
        f.write(s)
    model = fasttext.train_unsupervised('word_vectors/transcript_text.txt', minn=2, maxn=6, dim=300)
    model.save_model('word_vectors/word_vectors.bin')
    breakpoint()


if __name__ == '__main__':
    main()
