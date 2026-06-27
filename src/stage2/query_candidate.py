# query 1 ảnh đầu vào ngẫu nhiên -> cho ra 1600 ảnh candidate
import json


def query_candidate(input_image, dataset_name='roxford5k_self_pairs', num_candidates=1600):
    with open(f"src/stage2/{dataset_name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    table = {
        q["query_name"]: q["results"]
        for q in data["queries"]
    }
    return table[input_image][:num_candidates]

# if __name__ == "__main__":
#     query_candidate("all_souls_000013")

