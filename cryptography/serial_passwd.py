import sys
def rc4(strs:str):
    init_key = sys.stdin.readline()
    while init_key:
        lens = len(init_key)
        S, T = [0] * 256, [0] * 256
        # init list
        for i in range(256):
            S[i] = i
            T[i] = init_key[i%lens]

        j = 0
        for i in range(256):
            j = (j + S[i] + ord(T[i]))%256
            S[i], S[j] = S[j], S[i]

        # generate key
        i, j = 0, 0
        len_m = len(strs)
        key = []
        while len(key) < len_m:
            i = (i + 1)%256
            j = (j + S[i])%256
            S[i], S[j] = S[j], S[i]
            t = (S[i] + S[j])%256
            key.append(S[t])
        print(key)
        for i in range(len_m):
            print(chr(key[i] ^ ord(strs[i])))
        init_key = sys.stdin.readline()

if __name__ == '__main__':
    rc4('中国')