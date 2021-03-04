지난 python 정리 글에 이어 이번에는 Numpy를 공부해보려고 한다. 

# Numpy

파이썬이 계산과학분야에 이용될 때 주요 역할을 하는 라이브러리이다. 고성능의 다차원 배열 객체를 다룰 때 사용된다.

### 배열

동일한 자료형을 가지는 값들이 행렬 형태로 있는 것이다. 각각의 값들은 튜플 형태로 인덱싱 된다.

- rank : 배열이 몇 차원인지를 의미한다.
- shape : 각 차원의 크기를 알려주는 정수들으 모인 튜플이다.

``` python
import numpy as np

a = np.array([1,2,3])   # rank가 1인 배열 생성
print(type(a))          # 출력 "<type 'numpy.ndarray'>"
print(a.shape)          # 출력 "(3,)"
print(a[0], a[1], a[2]) # 출력 "1 2 3"
a[0] = 5
print(a)

b = np.array([[1,2,3],[4,5,6]]) # rank가 2인 배열 생성
print(b.shape)                  # 출력 "(2,3)"
print(b[0,0], b[0,1], b[1,0])   # 출력 "1 2 4"
```

다양한 함수를 제공하는 Numpy

```python
import numpy as np

a = np.zeros((2,2)) # 모든 값이 0인 배열 생성
print(a)            # [[0. 0.]
                    # [0. 0.]]

b = np.ones((1,2))  # 모든 값이 1인 배열 생성
print(b)            # [[1. 1.]]

c = np.full((2,2), 7)   # 모든 값이 특정 상수인 배열 생성
print(c)            # [[7 7]
                    # [7 7]]

d = np.eye(2)       # 2 x 2 단위행렬(unit matrix) 생성
print(d)            # [[1. 0.]
                    # [0. 1.]]

e = np.random.random((2,2))     # 임의의 값
print(e)                        # [[0.15524346 0.01775175]
                                # [0.48968497 0.50222192]]
```



### 배열 인덱싱

**슬라이싱**

```python
import numpy as np

# rank : 2 / shape (3,4)
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

b = a[:2, 1:3] # (첫 두행(0행, 1행))과, (1열,2열)로 이루어진 부분 배열

# 슬라이싱된 배열은 원본 배열과 같은 데이터를 참조한다. 
# 슬라이싱된 배열을 수정하면 원본 배열 역시 수정된다.
print(a[0,1])   # 출력 2
b[0,0] = 77     # b[0,0]은 a[0,1]과 같은 데이터이다.
print(a[0,1])   # 출력 "77"
```

정수를 이용한 인덱싱과 슬라이싱을 혼합해서 사용 할 때, 기존의 배열보다 낮은 rank의 배열이 얻어짐을 확인할 수 있다.

```python
import numpy as np

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# 배열의 중간 행에 접근하는 방법
# 1. 정수 인덱싱과 슬라이싱을 혼합해서 낮은 rank의 배열 생성된다.
# 2. 슬라이싱만 사용해서 원본 배열과 동일한 rank의 배열이 생성된다.
row_r1 = a[1,:]     # 배열 a의 두 번째 행을 rank가 1인 배열로
row_r2 = a[1:2, :]  # 배열 a의 두 번째 행을 rank가 2인 배열로
print(row_r1, row_r1.shape) # [5 6 7 8] (4,)
print(row_r2, row_r2.shape) # [[5 6 7 8]] (1, 4)
```

**정수 배열 인덱싱**

```python
import numpy as np

a = np.array([[1,2], [3,4], [5,6]])

print(a[[0,1,2], [0,1,0]])  # 출력 "[1 4 5]"
print(np.array([a[0,0], a[1,1], a[2,0]]))   # 출력 "[1 4 5]"

# 정수 배열 인덱싱을 사용 할 때, 원본 배열의 같은 요소를 재사용할 수 있음.
print(a[[0,0], [1,1]])               # 출력 "[2, 2]"
print(np.array([a[0,1], a[0,1]]))    # 출력 "[2, 2]"
```

```python
import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9],[10,11,12]])
print(a)

b = np.array([0,2,0,1])

print(a[np.arange(4),b])    # 출력 [ 1  6  7 11] (arange(4)를 통해서 -> 0,1,2,3)

# b에 저장된 인덱스를 이용해 각 행에서 하나의 요소를 변경한다.
a[np.arange(4),b] += 10
print(a)
```

**불리언 배열 인덱싱**

배열 속 요소 중 필요한 것만 선택할 수 있다. 특정 조건을 만족하게 하는 요소만 선택하고자 할 때 사용된다.

```python
import numpy as np

a = np.array([[1,2], [3,4], [5,6]])

bool_idx = (a > 2)  # 2보다 큰 a의 요소를 찾을 수 있다. 
print(bool_idx)     # 출력 "[[False False]
                    #       [True   True]
                    #       [True   True]]"

print(a[bool_idx])  # 출력 "[3 4 5 6]"

# 위의 과정을 정리하면
print(a[a>2])
```

### 자료형

- Numpy는 배열이 생성될 때 자료형을 스스로 추측한다.
- 배열을 생성할 때 명시적으로 특정 자료형을 지정할 수도 있다.

```python
import numpy as np
x = np.array([1,2])	# Numpy가 자료형을 추측해서 선택
x = np.array([1,2], dtype=np.int64)		#	특정 자료형을 명시적으로 지정
print(x.dtype)
```

### 배열 연산

- 수학 함수는 배열의 각 요소별로 동작하며 연산자를 통해 동작하거나 numpy 함수모듈을 통해 동작한다.
- '*'은 행렬 곱이 아닌 요소별 곱이다. 
- 벡터의 내적, 벡터와 행렬의 곱, 행렬곱을 위해 'dot'함수를 사용한다.

```python
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11,12])

# 벡터의 내적 : 결과는 모두 219
print(v.dot(w))
print(np.dot(v,w))

# 행렬과 벡터의 곱; 둘 다 결과는 rank 1인 배열 [29 67]
print(x.dot(v))
print(np.dot(x,v))

# 행렬곱
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x,y))
```

**sum**

**sum**을 이용하여 연산하기

```python
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))    # 모든 요소를 합한 값 "10"
print(np.sum(x,axis=0)) # 각 열에 대한 합을 연산 "[4 6]"
print(np.sum(x,axis=1)) # 각 행에 대한 합을 연산 "[3 7]"
```

**Transpose Matrix (전치행렬)** 

'T' 속성을 사용하여 행렬의 주 대각선을 기준으로 대칭되는 요소를 바꾸기 

```python
import numpy as np

x = np.array([[1,2],[3,4]])
print(x)
# [[1 2]
#  [3 4]]

print(x.T)
# [[1 3]
#  [2 4]]
```

### 브로트캐스팅

Numpy에서 shape가 다른 배열 간에도 산술 연산이 가능하게 하는 매커니즘을 말한다. 

큰 배열과 작은 배열이 있을 때, 큰 배열ㅇ르 대상으로 작은 배열을 여러 번 연산하고 할 때 사용된다.

```python
import numpy as np

# 행렬 x의 각 행에 벡터 v를 더한 뒤,
# 그 결과를 행렬 y에 저장한다.

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v = np.array([1,0,1])
y = np.empty_like(x)    # x와 동일한 shape를 가지며 비어있는 행렬 생성

for i in range(4):
    y[i, :] = x[i, :] + v

print(y)

# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
```

'x'가 매우 큰 행렬이라면, 파이썬의 반복문을 통한 코드는 매우 느리다. 

벡터 'v'를 행렬 'x'의 각 행에 더하기 위해 'v'를 여러 개 복사해 주식으로 쌓은 행렬을 만들어서 더한다.

```python
import numpy as np

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v = np.array([1,0,1])
vv = np.tile(v, (4,1))  # v의 복사본 4개를 위로 올린다.
print(vv)
# [[1 0 1]
#  [1 0 1]
#  [1 0 1]
#  [1 0 1]]

y = x + vv      # x와 vv의 각 요소들의 합
print(y)
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]

# 브로드캐스팅을 이용하여 v를 x의 각 행에 더하기
yy = x + v
print(yy)
```

**배열의 브로드캐스팅 규칙**

1. 두 배열이 동일한 rank를 가지고 있지 않다면, 낮은 rank의 1차원 배열이 높은 rank 배열의 높은 rank 배열의 shape로 간주한다.
2. 특정 차원에서 두 배열이 동일한 크기를 갖거나, 두 배열 중 하나의 크기가 1이라면 두 배열은 특장 차원에서 compatible (서로 호환가능)하다고 여겨진다.
3. 두 행렬이 모든 차원에서 compatible하다면, 브로드캐스팅이 가능하다.
4. 브로드캐스팅이 이뤄지면, 각 배열 shape의 요소별 최소공배수로 이루어진 shape가 두 배열의 shape로 간주한다.
5. 차원에 상관없이 크기가 1인 배열과 1보다 큰 배열이 있을 때, 크기가 1인 배열은 자신의 차원 수만큼 복사되어 쌓인 것처럼 간주한다. 

```python
import numpy as np

v = np.array([1,2,3])
w = np.array([4,5])

# 외적을 계산한다. 
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v,(3,1))*w)

# 벡터를 행렬의 각 행에 더 하는 경우
# [[2 4 6]
#  [5 7 9]]
x = np.array([[1,2,3],[4,5,6]])
print(x+v)

# 벡터를 행렬의 각 행에 더하기
# x는 shape가 (2,3)이고 w는 shape가 (2,)일 때
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)

# x의 shape는 (2, 3)이다. Numpy는 스칼라를 shape가 ()인 배열로 취급한다.
print(x + np.reshape(w, (2,1)))
```



