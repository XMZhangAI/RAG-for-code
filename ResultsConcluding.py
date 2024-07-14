'''将 result 中的文件，参照 zxm 的一系列 conclude 文件进行统计，得到一个整体性的报告'''
'''最后、与 zxm 所做的工作进行比较'''
'''这里只是做数据和文件的转换工作'''
import json

def process_data(file_path):
    # Initialize dictionaries to store total scores and counts for averaging
    scores = {'exact_match':0,'edit_similarity': 0.0, 'identifier_match': 0.0, 'count': 0}
    # Read the file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            # Check the context_type and update the corresponding scores and counts
            scores['exact_match']+=data['score']['exact_match']
            scores['edit_similarity'] += data['scores']['edit_similarity']
            scores['chunk']['identifier_match'] += data['scores']['identifier_match']
            scores['count'] += 1


    # Calculate averages
    averages = {}
    if scores['count'] > 0:
        averages = {
            'exact_match_rate':scores['exact_match']/ scores['count'],
            'edit_similarity_avg': scores['edit_similarity'] / scores['count'],
            'identifier_match_avg': scores['identifier_match'] / scores['count']
        }
    else:
        averages = {'exact_match_rate':None,'edit_similarity_avg': None, 'identifier_match_avg': None}

    return averages

# Specify the path to your JSONL file
file_path = ''

# Process the file and get the averages
averages = process_data(file_path)
print(averages)

