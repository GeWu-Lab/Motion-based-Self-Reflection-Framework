import json
import h5py
import os

if __name__ == "__main__":
    data_dir = "adjust_llava_motion"
    data_path = "adjust_llava_motion/all_dataset_mapping.json"
    json_data = json.load(open(data_path, "r"))
    
    datasets = json_data['dataset']
    language_set = set()
    for data_name in datasets:
        path = os.path.join(data_dir, data_name + "_adjust_llava_motion.hdf5")
        f = h5py.File(path, 'r')
        for i in range(1,501):
            languages = f['data'][f'demo_{i}']['language'][:]
            for language in languages:
                language = language.decode('utf-8')
                language_set.add(language)
    
    print("the language set is:", language_set)
    print("the language set is:", len(language_set))
    language_list = sorted(list(language_set))
    language_dict = {}
    for idx, l in enumerate(language_list):
        language_dict[l] = idx
    new_json = {}
    new_json['dataset'] = datasets
    new_json['language_idx'] = language_dict
    
    with open("adjust_llava_motion/language_idx.json", "w") as f:
        json.dump(new_json, f)