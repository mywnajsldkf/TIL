- Java, C++ 베이스에서 파이썬으로 옮겼을 때 실수하기 쉬운 @classmethod와 @staticmehod의 차이를 알아보자.



### @staticmethod

> 정적메소드

```python
class Test:
  num = 0
  
  @staticmethod
  def add(x,y):
    return x+y
  
print Test.add(1,1)
```

클래스에서 직접 접근하여 함께 공유한다.



> 객체를 통해서 정적함수를 호출하는 코드

```python
class Test:
  num = 0
  
  @staticmethod
  def add(x,y):
    return x+y
  
t = Test()
print t.add(1,1)
```



> 정적함수이므로 self를 통한 접근은 불가능하다. 아래가 해당 사례

```python
class Test:
  num = 10
  
  @staticmethod
  def add(x,y):
    return x+y+self.num
  
t = Test()
print t.add(1,1)
```



### @classmethod

```python
class Test:
  num = 10
  
  @classmethod
  def add(x,y):
    return x + y
  
print Test.add(1,1)
```

이렇게 하면 에러



> 클래스 메소드는 클래스 자체를 첫번째 인자로 넣어줘야한다. (클래스 자체를 객체로 봄)

```python
class Test:
  num = 10
  
  @classmethod
  def add(cls,x,y):
    return x+y
  
print Test.add(1,1)
```

인자로 cls말고 다른 말로 넣어도 된다.



### 클래스 method가 왜 필요할까?

> @staticmethod를 이용해 method를 만들었을 때

```python
class Date:
  word = 'date :'
  
  def __init__(self, date):
    self.date = self.word + date
    
  @staticmethod
  def now():
    return Date("today")
  
  def show(self):
    print self.date
    
a = Date("2016, 9, 13")
a.show()
b = Date.now()
b.show()
```

지정일/오늘로 초기화한 객체를 돌려준다.

date : 2016, 9, 13

date : today

> 상속받아서 진행할 때

```python
class KoreanDate(Date):
  word = '날짜:'
  
a = KoreanDate.now()
a.show()
```

now() 를 통해 KoreanDate객체가 생성되는게 아닌 Date객체가 생성된다.

date : today

> @classmethod를 사용할 때

```python
class Date:
  word = 'date: '
  def __init__(self, date):
    self.date = self.word + date
    
  @classmethod
  def now(cls):
    return cls("today")
  
  def show(self):
    print self.date
    
class KoreanDate(Date):
  word = '날짜:'
  
a = KoreanDate.now()
a.show()
```

now를 @classmethod로 바꾸어서 cls를 전달해서 그것으로 객체를 생성하면 Date객체 대신에 KoreanDate객체를 돌려받는다.

날짜 : today



> ref

https://hamait.tistory.com/635

