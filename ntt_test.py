import random
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


# def ntt_prime(bits: int, n: int) -> int:
#     """
#     Finds a prime `q` suitable for an NTT of order `n` with a specified bit length.
#     This function utilizes features from the `galois` library.

#     It searches for a prime `q` of the form `q = 2nk + 1`.

#     Inputs:
#         bits (int): The desired number of bits for the prime (e.g., 120).
#         n (int): The order of the NTT (e.g., 2048).

#     Output:
#         int: The prime `q` that satisfies the conditions.
#     """
#     if not isinstance(bits, int) or bits <= 1:
#         raise ValueError(f"'bits' must be an integer greater than 1. Received: {bits}")
#     if not isinstance(n, int) or n <= 0:
#         raise ValueError(f"'n' must be a positive integer. Received: {n}")

#     divisor = 2 * n

#     # 1. Select a random starting point for the search.
#     # This ensures that a different prime can be found on each run.
#     lower_bound_q = 2**(bits - 1)
#     upper_bound_q = 2**bits
    
#     # Starting the search near a multiple of the divisor is more efficient.
#     # The starting point itself does not need to be prime.
#     start = random.randint(lower_bound_q, upper_bound_q)
#     q = (start // divisor) * divisor + 1
#     if q < start:
#         q += divisor
    
#     # 2. Use galois.next_prime() to find the next prime and check if it meets the criteria.
#     while True:
#         # First, check if the current `q` is within the valid range and is a prime that satisfies the condition.
#         if q < upper_bound_q and galois.is_prime(q):
#             if (q - 1) % divisor == 0:
#                 return q
        
#         # Find the next prime candidate.
#         q = galois.next_prime(q)

#         # If the search exceeds the bit range, raise an exception.
#         if q >= upper_bound_q:
#             raise RuntimeError(f"Could not find a suitable prime within the {bits}-bit range.")

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
    q = galois.ntt_prime(120, n)

    # 아래는 n = 2**4 일 때의 q
    # q = 698018593
    
    # 아래는 n = 2**3 일 때의 q
    # q = 607613921
    GF = galois.GF(q)

    a = [1 for _ in range(n)]
    b = [5 for _ in range(n)]

    start_time = time.time()
    ntt_a = galois.ntt(x=GF(a), size=n)
    # print("NTT a : ", ntt_a)
    end_time = time.time()
    print(f"NTT of a took {end_time - start_time:.6f} seconds")
    
    
    ntt_b = galois.ntt(x=GF(b), size=n)
    mul_res = [a * b for a, b in zip(ntt_a, ntt_b)]
    intt_mul_res = galois.intt(X=GF(mul_res), size=n).tolist()
    intt_mul_res = balanced_mod_vector(intt_mul_res, q)
    print(intt_mul_res[:10])