Yeonsik-school을 본격적으로 시작하기전에 공부 python에 대한 개념을 정리해보자.

이 글은 http://aikorea.org/cs231n/python-numpy-tutorial/#python을 바탕으로 실습을 진행했음을 밝힌다.

# 파이썬

### 기본 자료형

1. 숫자

   - 정수형(Integers)과 실수형(floats) 데이터 타입이 동일한 역할을 한다.
   - 증감 단항연산자(`x++`, `x--`)는 존재하지 않는다.
   - long 정수형과 복소수 데이터 타입이 구현되어 있다.

2. 불리언(Booleans)

   - 논리 자료형의 모든 연산자가 구현되어 있다.
   - 다만 기호(`&&`, `||`, 등)은 영어 단어로 구현되어 있다.

3. 문자열

   - 파이썬은 문자열과 다양한 기능을 지원한다.

     ```python
     hello = 'hello'     # String 문자열을 포현할 때 따옴표나
     world = "world"     # 쌍따옴표가 사용된다. 어떤 것을 사용해도 괜찮다.
     print (hello)         # 출력 "hello"
     print (len(hello))    # 문자열 길이 출력 : "5"
     hw = hello + ' ' + world    # 문자열 연결
     print (hw)            # 출력 "hello world"
     hw12= '%s %s %d % (hello, world, 12)'   # sprintf 방식의 문자열 서식 지정
     print (hw12)
     ```

     [문자열 메소드 더 알아보기](https://docs.python.org/2/library/stdtypes.html#string-methods)

### 컨테이너

1. 리스트

   - 파이썬에서 배열 같은 존재이다.
   - 배열과 달리 크기 변경이 가능하고 서로 다른 자료형일지라도 하나의 리스트에 저장될 수 있다.

   ```python
   xs = [3,1,2]        # 리스트 생성
   print(xs, xs[2])    # 출력 "[3,1,2] 2"
   print(xs[-1])       # 인덱스가 음수일 경우 리스트의 끝에서 세어짐; 출력 "2"
   xs[2] = 'foo'       # 리스트는 자료형이 다른 요소들을 저장할 수 있다.
   print(xs)           # 출력 "[3,1,'foo']"
   xs.append('bar')    # 리스트의 끝에 새 요소 추가
   print(xs)           # 출력 "[3,1,'foo', 'bar']"
   x = xs.pop()        # 리스트의 마지막 요소 삭제 후 반환
   print(x, xs)        # 출력 "bar [3,1, 'foo']"
   ```

   **슬라이싱**

   리스트의 일부분에만 접근하는 간결한 문법

   ```python
   nums = range(5)     # 정수들로 구성된 리스트를 만든다.
   print(nums)         # 출력 "[0, 1, 2, 3, 4]"
   print(nums[2:4])    # 인덱스 2에서 4(제외)까지 슬리이싱; 출력 "[2,3]"
   print(nums[2:])     # 인덱스 2에서 끝까지 슬라이싱; 출력 "[2,3,4]"
   print(nums[:2])     # 처음부터 인덱스 2(제외)까지 슬라이싱; 출력 "[0,1]"
   print(nums[:])      # 전체 리스트 슬라이싱; 출력 ["0,1,2,3,4"]
   print(nums[:-1])    # 슬라이싱 인덱스는 음수도 가능; 출력 ["0,1,2,3"] ('4'를 제외하고)
   nums[2:4] = [8,9]   # 슬라이스된 리스트에 새로운 리스트 할당
   print(nums)         # 출력 "[0,1,8,9,4]"
   ```

   **반복문**

   리스트의 요소들을 반복해서 조회할 수 있다. 한줄씩 출력됨을 확인할 수 있다.

   ```python
   animals = ['cat', 'dog', 'monkey']
   for animal in animals:
       print(animal)
   ```

   반복문 내에서 리스트 각 요소의 인덱스에 접근하려고 할 때 'enumerate'함수를 사용한다.

   ```python
   animals = ['cat', 'dog', 'monkey']
   for idx, animal in enumerate(animals):
     print '#%d: %s' % (idx + 1, animal)
   ```

   **리스트 자료형 변환**

   1. 숫자의 제곱을 계산하는 코드를 작성하자.

      ```python
      nums = [0,1,2,3,4]
      squares = []
      for x in nums:
        squares.append(x ** 2)
      print(squares)	# 출력 [0,1,4,9,16]
      ```

   2. 더 간단하게 해 보자.

      ```python
      nums = [0,1,2,3,4]
      squares = [x ** 2 for x in nums]
      print(squares)
      ```

   3. 조건 추가

      ```python
      even_squares = [x ** 2 for x in nums if  x % 2 == 0]
      ```

2. 딕셔너리

   {key, value} 쌍을 저장한다.

   ```python
   d = {'cat':'cute', 'dog':'furry'}		# 새로운 딕셔너리를 만든다.
   print (d['cat'])        # 딕셔너리의 값을 받음; 출력 "cute"
   print ('cat' in d)      # 딕셔너리가 주어진 key를 가지고 있는지 확인; 출력 "True"
   d['fish'] = 'wet'       # 딕셔너리 값 지정
   print (d.get('fish'))   # 출력 : wet
   del d['fish']           # 딕셔너리에 저장된 요소 삭제
   ```

   **반복문**

   ```python
   d = {'person':2, 'cat': 4, 'spider': 8}
   for animal in d:
       legs = d[animal]
       print('A %s has %d legs' % (animal, legs))
   ```

3. 집합

   순서 구분이 없고 서로 다른 요소 간의 모임

   ```python
   animals = {'cat', 'dog'}
   print 'cat' in animals      # 출력 : 'True'
   animals.add('fish')         # 요소를 집합에 추가
   print (len(animals))        # 집합에 포함된 요소의 수; 출력 "3"
   animals.remove('cat')       # 요소를 집합에서 제거
   print(len(animals))         # 출력 "2"
   ```

4. 튜플

   요소 간 순서가 있으며 값이 변하지 않는 리스트

   변수 값을 바꾸는 것을 해보자.

   **temp를 사용하는 경우**

   ```python
   a = 10
   b = 20
   temp = a # a값을 temp에 저장 (temp = 10)
   a = b    # b값을 a에 저장    (a = 20)
   b = temp # temp값을 b에 저장 (b = 10)
   print(a,b)  # 20, 10
   ```

   **튜플이 파이썬에는 있다!**

   ```python
   c = 10
   d = 20
   c,d = d,c
   print (c,d)  # 20 10
   ```

   세 번째 줄에서 등호 왼쪽은 c,d 라는 변소가 담긴 튜플이다.

   오른쪽은 d와 c의 값이 담긴 튜플이다.

   1. 튜플은 리스트와 달리 원소값을 직접 바꿀 수 없으므로, slice를 이용한다.

      ```python
      p = (1,2,3)
      q = p[:1] + (5,) + p[2:]
      print(q)                    # 출력 : (1, 5, 3)
      r = p[:1], 5, p[2:]
      print(r)                    # 출력 : ((1,), 5, (3,))
      ```

   2. 튜플을 리스트로, 리스트를 튜플로 바꿀 수도 있다.

      ```python
      p = (1,2,3)
      q = list(p)
      print(q)        # [1, 2, 3]
      r = tuple(q)
      print(r)        # (1, 2, 3)
      ```

### 함수

`def` 키워드를 통해 정의된다.

```python
def hello():
  print("hello!")
```

### 클래스

```python
class Greeter(object):
    # 생성자
    def __init__(self, name):
        self.name = name    # 인스턴스 변수 선언

    # 인스턴스 메소드
    def greet(self, loud=False):
        if loud:
            print("Hello, %s!" % self.name.upper())
        else:
            print("Hello, %s" % self.name)
    
g = Greeter('Fred')
g.greet()
g.greet(loud=True)
```

