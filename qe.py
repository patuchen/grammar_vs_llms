# qual_est.py
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import argparse
import os


class QualityEstimator:
    def __init__(self, model, data_path, source_language, target_language, batch_size):
        model_path = download_model(model)
        self.model = load_from_checkpoint(model_path)
        self.data_path = data_path
        self.source_language = source_language
        self.target_language = target_language
        self.batch_size = batch_size

        self.src_path = f"{self.data_path}/{self.source_language}-{self.target_language}/test-src.txt"
        self.ref_path = f"{self.data_path}/{self.source_language}-{self.target_language}/test-ref.txt"

        self.hyp_path = f"{self.data_path}/{self.source_language}-{self.target_language}/test-sys.txt"
        self.ped_path = f"{self.data_path}/{self.source_language}-{self.target_language}/test-sys-ped.txt"
        self.qe_ped_path = f"{self.data_path}/{self.source_language}-{self.target_language}/test-sys-ped-qe.txt"


    def qual_est(self):

        src_sentences = _file2list(self.src_path)
        
        hyp_sentences = _file2list(self.hyp_path) if os.path.exists(self.hyp_path) else None
        ref_sentences = _file2list(self.ref_path) if os.path.exists(self.ref_path) else None

        qual_est = []

        ref_inputs = self._construct_input(src_sentences, ref_sentences)
        hyp_inputs = self._construct_input(src_sentences, hyp_sentences) if hyp_sentences else None

        for i in tqdm(range(0, len(src_sentences), args.batch_size)):

            # score references
            ref_data = ref_inputs[i:i+args.batch_size]
            ref_model_output = self.model.predict(ref_data, batch_size=args.batch_size, gpus=1)[0]

            # score hypotheses, if we have them
            hyp_data = hyp_inputs[i:i+args.batch_size] if hyp_inputs else None
            if hyp_data:
                hyp_model_output = self.model.predict(hyp_data, batch_size=args.batch_size, gpus=1)[0]

            qual_est.extend(ref_model_output)

            
        ref_qual_est = qual_est
        _list2file(ref_qual_est, f"{self.data_path}/{self.source_language}-{self.target_language}/test-ref-qe_scores.txt")

        # sorting
        zipped = list(zip(src_sentences, ref_sentences, ref_qual_est))
        sorted_zip = sorted(zipped, key=lambda x: x[2], reverse=True)
        sorted_src = [item[0] for item in sorted_zip]
        sorted_ref = [item[1] for item in sorted_zip]
        sorted_qual_est = [item[2] for item in sorted_zip]
        _list2file(sorted_src, f"{self.data_path}/{self.source_language}-{self.target_language}/src-sorted.txt")
        _list2file(sorted_ref, f"{self.data_path}/{self.source_language}-{self.target_language}/ref-sorted.txt")
        _list2file(sorted_qual_est, f"{self.data_path}/{self.source_language}-{self.target_language}/scores-sorted.txt")

    def _construct_input(self, src_sentences, hyp_sentences):
        return [
            {
                "src": src_sentence,
                "mt": hyp_sentence
            }
            for src_sentence, hyp_sentence in zip(src_sentences, hyp_sentences)
        ]


    def _evaluate(self, src_path, ref_path, hyp_path, bleu_path, comet_path):
        tok="13a" if self.target_language != "zh" else "zh"

        os.system(f"SACREBLEU_FORMAT=text sacrebleu -m bleu chrf -tok {tok} -w 2 {ref_path} < {hyp_path} > {bleu_path}")
        os.system(f"cat {bleu_path}")

        # os.system(f"comet-score -s {source-input}.txt -t {translation-output}.txt --model Unbabel/wmt22-cometkiwi-da --gpus 1 > {qe_path}")

        os.system(f"comet-score --quiet -s {src_path} -t {hyp_path} -r {ref_path} --batch_size 256 --model Unbabel/wmt22-comet-da --gpus 1 > {comet_path}")
        os.system(f"tail -n 1 {comet_path}")


def _list2file(string_list, file_path):
    with open(file_path, 'w') as file:
        for string in string_list:
            file.write(f"{string}\n")

def _file2list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Unbabel/wmt22-cometkiwi-da")
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--src", type=str, default="en")
    parser.add_argument("--tgt", type=str, default="zh")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    qualest = QualityEstimator(
        model=args.model,
        data_path=args.data_path,
        source_language=args.src,
        target_language=args.tgt,
        batch_size=args.batch_size,
    )

    qualest.qual_est()
