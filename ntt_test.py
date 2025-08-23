import galois


# 예제 사용
if __name__ == "__main__":
    n = 2**3
    q = galois.ntt_prime(bits=120, n=n)

    # 아래는 n = 2**4 일 때의 q
    # q = 698018593
    
    # 아래는 n = 2**3 일 때의 q
    # q = 607613921
    GF = galois.GF(q)

    a = [1 for _ in range(n)]
    b = [5 for _ in range(n)]

    
    ntt_a = galois.ntt(x=GF(a), size=n)
    ntt_b = galois.ntt(x=GF(b), size=n)
    mul_res = [a * b for a, b in zip(ntt_a, ntt_b)]
    intt_mul_res = galois.intt(X=GF(mul_res), size=n).tolist()
    intt_mul_res = galois.balanced_mod_vector(intt_mul_res, q)
    print(intt_mul_res[:10])