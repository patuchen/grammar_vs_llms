import subset2evaluate.utils

def load_data(split="tiny_test", langs="three"):
    import random
    import warnings

    data = {
        args: subset2evaluate.utils.load_data_wmt(args[0], args[1])
        for args in [
            ("wmt24", "cs-uk"),
            ("wmt24", "en-cs"),
            ("wmt24", "en-de"),
            ("wmt24", "en-es"),
            ("wmt24", "en-hi"),
            ("wmt24", "en-is"),
            ("wmt24", "en-ja"),
            ("wmt24", "en-ru"),
            ("wmt24", "en-uk"),
            ("wmt24", "en-zh"),
            ("wmt24", "ja-zh"),
        ]
    }

    if langs == "three":
        data = {
            ("wmt24", langs): data[("wmt24", langs)]
            for langs in ["cs-uk", "en-de", "en-zh"]
        }
    elif langs != "all":
        if ("wmt24", langs) not in data:
            raise ValueError(f"Language pair {langs} not found in WMT24")
        data = {("wmt24", langs): data[("wmt24", langs)]}

    for k in data:
        random.Random(0).shuffle(data[k])

    if split == "micro_test":
        sample_size = 10
    elif split == "tiny_test":
        sample_size = 100
    elif split == "test":
        sample_size = 500
    elif split == "all":
        sample_size = None
        warnings.warn("Using all data which is not aligned with any of the experiments.")
    
    data_new = []
    for k in data:
        for line in data[k]:
            line["langs"] = k[1]
        data_new += data[k][:sample_size]

    return data_new