import random
import secrets
import time
from typing import List
import gmpy2
import galois

import numpy as np
import galois # 실제 실행을 위해 galois 라이브러리가 필요합니다.
        
def balanced_mod(num: int, q: int) -> int:
    """
    Returns the balanced modulus of a number in Z_q.
    The result is in the range of [-q/2, q/2).
    """
    if num >= q // 2:
        return num - q
    return num

def balanced_mod_vector(vec: List[int], q: int) -> List[int]:
    """
    Applies balanced modulus to each element of the vector `vec`.
    """
    return [balanced_mod(x, q) for x in vec]

def find_ntt_prime(bits, n):
    """

    
    :param bits: 원하는 소수의 비트 수 (예: 120)
    :param n: NTT의 차수 (예: 2048)
    :return: 조건을 만족하는 소수 q (gmpy2.mpz 타입)
    """
    
    divisor = 2 * n
    
    # 1. k의 범위를 계산합니다.
    lower_bound_q = gmpy2.mpz(2)**(bits - 1)
    upper_bound_q = gmpy2.mpz(2)**bits
    
    lower_bound_k = (lower_bound_q - 1) // divisor
    upper_bound_k = (upper_bound_q - 1) // divisor
    
    print(f"Finding a {bits}-bit prime q such that (q-1) is a multiple of {divisor}...")
    print(f"Searching for k in range [{lower_bound_k}, {upper_bound_k}]")
    
    # 2. 소수를 찾을 때까지 k를 무작위로 선택하고 테스트합니다.
    while True:
        # secrets.randbelow는 정수 상한값을 받아 0부터 상한값-1까지의 난수를 생성
        k_range = upper_bound_k - lower_bound_k
        random_offset = secrets.randbelow(k_range)
        k = lower_bound_k + random_offset
        
        # 후보 소수 q 계산
        q = k * divisor + 1
        
        # 3. gmpy2.is_prime()으로 소수인지 판별합니다.
        # 이 함수는 밀러-라빈 테스트를 사용하며, 매우 신뢰도가 높습니다.
        if gmpy2.is_prime(q):
            print(f"\nFound prime! q = {q}")
            return int(q)

def fast_remainder_negacyclic(p, n):
    """
    Quickly computes the remainder when polynomial p(x) is divided by x^n + 1.
    
    :param p: Coefficients of a polynomial of degree 2n-1. numpy array. (length is 2n)
              [p_0, p_1, ..., p_{2n-1}] (from lowest degree)
    :param n: The n in the divisor polynomial x^n + 1
    :return: Coefficients of the remainder. numpy array. (length is n)
    """
    if len(p) != 2 * n:
        raise ValueError("The length of input polynomial coefficients must be 2n.")
    
    # Split polynomial coefficients into lower and upper parts
    p_low = p[:n]  # p_0, ..., p_{n-1}
    p_high = p[n:] # p_n, ..., p_{2n-1}
    
    # Remainder is (lower part) - (upper part)
    remainder = [a - b for a, b in zip(p_low, p_high)]
    
    return remainder

# 예제 사용
if __name__ == "__main__":
    n = 2**3
    q = find_ntt_prime(30, n)
    GF = galois.GF(q)
    
    a = [1, 1, 1, 1, 1, 1, 1, 1]
    b = [5, 5, 5, 5, 5, 5, 5, 5]
    
    start_time = time.time()
    ntt_a = galois.ntt(x=GF(a), size=n)
    end_time = time.time()
    print(f"NTT of a took {end_time - start_time:.6f} seconds")
    ntt_b = galois.ntt(x=GF(b), size=n)
    mul_res = [a * b for a, b in zip(ntt_a, ntt_b)]
    intt_mul_res = galois.intt(X=GF(mul_res), size=n).tolist()
    intt_mul_res = balanced_mod_vector(intt_mul_res, q)
    print(intt_mul_res)