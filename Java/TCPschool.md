# 1. 자바 시작

#### 자바 프로그래밍

<img src="http://www.tcpschool.com/lectures/img_java_programming.png" alt="img_java_programming" style="zoom:75%;" />

> **자바 컴파일러 (Java compiler)**

자바 소스 코드를 자바 가상 머신이 이해할 수 있는 자바 바이트 코드로 한다. 자바컴파일러는 자바 설치시 javac.exe 라는 파일 형태로 설치된다. 

> **자바 바이트 코드 (Java bytecode)**

자바 가상 머신이 이해할 수 있는 언어로 변환된 자바 소스 코드를 만한다. 확장자는 .class 이다.  자바 가상 머신만 있으면 어떤 운영체제라도 실행될 수 있다. 

> **자바 가상 머신(JVM)**

자바 바이트 코드를 실행시키기 위한 가상의 기계 

**구성**

1. 자바 인터프리터(interpreter)

   프로그램이 실행 중인 런타임에 실제 기계어로 변환해 주는 컴파일러

2. 클래스 로더(class loader)

   프로그램의 실행 속도를 향상시키기 위해 개발됨

3. JIT 컴파일러(Just-In-Time compiler)

   자바 컴파일러가 생성한 자바 바이트 코드를 런타임에 바로 기계어로 변환함

4. 가비지 컬렉터(garbage collector)

   더이상 사용하지 않는 메모리를 자동으로 회수함