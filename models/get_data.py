import json


def get_test_data():
    with open('./data/test/source.json') as source, open('./data/test/_source.json') as source_modified, open('./data/test/target.json') as target, open('./data/test/_target.json') as target_generated:
        return {
            "source": json.loads(source.read()),
            "source_modified": json.loads(source_modified.read()),
            "target": json.loads(target.read()),
            "target_generated": json.loads(target_generated.read()),
        }
