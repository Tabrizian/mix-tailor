from itertools import combinations

aggs = ['comed', 'krum', 'geomed', 'bulyan']
answers = []
for i in range(2, len(aggs) + 1):
    res = list(combinations(aggs, i))
    for result in res:
        answers.append(','.join(result))
print(answers)
