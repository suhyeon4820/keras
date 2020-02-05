from numpy import array
#(10, 4)
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):     # 10
        end_ix = i + n_steps           # 0 + 4 = 4   n_steps : 몇개씩 자르는지
        if end_ix > len(sequence)-1:   # 4 > 10-1
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] # 0, 1, 2, 3 / 4
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

# 0, 1, 2, 3 / 4
# 1, 2, 3, 4 / 5
# 2, 3, 4, 5 / 6
# 3, 4, 5, 6 / 7
# 4, 5, 6 ,7 / 8
# 5, 6, 7, 8 / 9

dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

n_steps = 3

x, y = split_sequence(dataset, n_steps)
print(x)
print('================================')
print(y)

