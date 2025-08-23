import time
import galois


# 예제 사용
if __name__ == "__main__":
    n = 2**3
    start_time = time.time()
    q = galois.ntt_prime(bits=120, n=n)
    end_time = time.time()
    print("소수 찾는 시간: ", end_time - start_time)

    # 아래는 n = 2**4 일 때의 q
    # q = 698018593
    
    # 아래는 n = 2**3 일 때의 q
    # q = 607613921
    GF = galois.GF(q)

    a = [1 for _ in range(n)]
    b = [5 for _ in range(n)]

    
    start_time = time.time()
    ntt_a = galois.ntt(x=GF(a), size=n)
    end_time = time.time()
    print("NTT 수행 시간: ", end_time - start_time)
    ntt_b = galois.ntt(x=GF(b), size=n)
    mul_res = [a * b for a, b in zip(ntt_a, ntt_b)]
    intt_mul_res = galois.intt(X=GF(mul_res), size=n).tolist()
    intt_mul_res = galois.balanced_mod_vector(intt_mul_res, q)
    print(intt_mul_res[:10])