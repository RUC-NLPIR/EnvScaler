"""Calculate average reward score from result file."""
import json


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def calc_avg_score(result_file_path):
    data = read_json(result_file_path)
    if len(data) > 0: 
        avg_score = round(sum([item["total_reward"] for item in data])*100/len(data), 2)
        all_num = len(data)
        print("result_file_path: ", result_file_path)
        print("all_num: ", all_num, "avg_score: ", avg_score)


if __name__ == "__main__":
    result_file_path = "your_result_file_path"
    calc_avg_score(result_file_path)