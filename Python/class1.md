python에서 자주 보이는 self가 잘 이해 가지 않았는데(또 직접 쓰려고 하니 막히는 부분이 있었음) 이 부분에 대하여 정리해보겠다.

이에 앞서 class에 대해 이해해야한다. 

## 클래스

### 절차지향 프로그래밍과 객체지향 프로그래밍

1. 절차지향 프로그래밍

   데이터와 데이터를 처리하는 함수가 분리되어 함수를 순차적으로 호출하여 데이터를 조작하는 방식

   ex) 고객 정보에 name, email, address라는 data를 관리해하는데 각각의 함수를 호출하여 하나하나 관리해야 하는 상황

   <img src="https://wikidocs.net/images/page/3454/6.01.png" alt="6.01" style="zoom:75%;"/>

2. 객체지향 프로그래밍

   - 객체를 정의한다.
   - 명함을 구성하는 데이터 : `이름`,`이메일`, `주소` 
   - 명함을 구성하는 함수 : 기본 명함을 출력하는 함수, 고급명함을 출력하는 함수

   <img src="https://wikidocs.net/images/page/3454/6.02.png" alt="6.02" style="zoom:75%;" />

   명함이라는 객체(타입)을 만들고 이를 활용하여 프로그래밍 하는 방식을 객체지향 프로그래밍이라고 한다. 



### 클래스 정의

```python
class BusinessCard:
  pass
```

클래스안에는 변수와 함수들이 포함되어 클래스를 정의한다. 

```python
card1 = BusinessCard()
card1
<__main__.BusinessCard object at 0x0302ABB0>
```

- BusinessCard : 파이썬에서 정의된 클래스

- card1 : 클래스를 이용해 생성된 인스턴스

  'BusinessCard'라는 클래스의 인스턴스가 메모리의 0x0302ABB0 위치에 생성되고 card1이라는 변수가 바인딩하게 된다.  



### 메서드

클래스를 이용해 생성된 인스턴스가 동작할 수 있도록 기능을 수행하는 함수 : 메서드(method)

```python
class BusinessCard:
  def set_info(self, name, email, addr):
    self.name = name
    self.email = email
    self.addr = addr
```

`set_info`라는 메서드를 정의하였다. 메서드를 정의할 때 함수와 같이 def 키워드를 사용한다. 

`set_info`메서드는 네  개개의 인자를 받는데, 그 중 name, email, addr은 사용자로부터 입력받은 데이터를 메서드로 전달할 때 사용하는 인자임을 말한다. 

**그렇다면 self는 무엇일까? **

*클래스 내부에 정의된 함수인 메서드의 첫 번째 인자는 반드시 `self`여야 한다.* 

메서드 인자로 전달된 값 name, email, addr을 self.name, self.email, self.addr라는 변수에 대입하는 것은 **바인딩**을 의미한다. (대입 -> 바인딩)

**클래스 인스턴스를 통해 메서드 호출하기**

```python
member1 = BusinessCard()
member1.set_info("Jeongin Kim", "mywnajsldkf@gmail.com","Korea")
```

`self.변수명`과 같은 형태를 띠는 변수를 인스턴스 변수라고 한다. 인스턴스 변수는 클래스 인스턴스 내부의 변수를 의미한다.

위 코드는 다음과 같다.

1. member1이라는 인스턴스를 생성한다.
2. `set_info ` 메서드를 호출하면 메서드의 인사로 전달된 값을 인스턴스 내부 변수은 self.name, self.email, self.addr이 바인딩(대입)한다. 
3. 클래스를 정의할 대 생성할 인스턴스 이름이 member1인지 모르므로 self라는 단어를 대신 사용한다. 

`self.변수명`은 나중에 생성될 클래스 인스턴스 내의 name 변수를 의미한다. 

**메서드 추가하기**

```python
class BusinessCard:
  def set_info(self, name, email, addr):
    self.name = name
    self.email = email
    self.addr = addr
  def print_info(self):
    print("===========")
    print("Name : ", self.name)
    print("E-mail : ", self.email)
    print("Address : ", self.addr)
    print("===========")
```



### 클래스 생성자

`__init__(self)`

클래스 인스턴스 생성과 초깃값 입력을 한번에 할 수 있는 방법

```python
class MyClass:
  def __init__(self):
    print("객체가 생성되었습니다.")
```

```python
class BusinessCard:
  def __init__(self, name, email, addr):
    self.name = name
    self.email = email
    self.addr = addr
  def print_info(self):
    print("------------")
    print("Name: ", self.name)
    print("E-mail: ", self.email)
    print("E-mail: ", self.addr)
    print("------------")
```

set_info 메서드가 데이터를 입력받는 역할을 수행했으므로 이를 `__init__`메서드로 이름만 변경하였다. 



### self는 무엇인가...

파이썬 내장 함수인 id를 이용해 인스턴스가 메모리에 할당된 주솟값을 확인해보자.

```python
class Foo:
  def func1():
    print("function 1")
  def func2(self):
    print(id(self))
    print("function 2")
```

func2메서드가 호출될 때 메서드의 인자로 전달되는 self의 id값을 화면에 출력하는 기능이다. 

**인스턴스를 생성한 후 해당 인스턴스의 메모리 확인하기**

```python
f = Foo()
id(f)
43219856
```

생성된 인스턴스 f가 메모리의 43219856 번지에 있다. 

**인스턴스 f를 이용해 func2 메서드 호출하기**

```python
f.func2()
43219856
function 2
```

주소 값이 같다.

***즉, 클래스 내에 정의된 self는 클래스 인스턴스이다!!!!!!***



다음 시간에는 클래스 네임스페이스, 클래스 변수와 인스턴스 변수, 상속에 대해 알아보겠다.