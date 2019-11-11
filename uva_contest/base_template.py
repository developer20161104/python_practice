import sys

# 在输入处理里面还有问题，需要处理
def find_marble():
    case = 1
    lists = sys.stdin.readline().strip('\n')
    while lists:
        lens, times = lists.split('')
        if lens == times == 0:
            break
        print("CASE# ", case)
        case += 1
        nums, guess = [], []
        for i in range(lens):
            nums.append(int(sys.stdin.readline().strip('\n')))
            if i < times:
                guess.append(int(sys.stdin.readline().strip('\n')))

        k = 0
        while times:
            if guess[k] in nums:
                print(guess[k], ' found at ', nums.index(guess[k]))
            else:
                print(guess[k], ' not found')
            k += 1
            times -= 1

    lists = sys.stdin.readline()


if __name__ == '__main__':
    find_marble()
