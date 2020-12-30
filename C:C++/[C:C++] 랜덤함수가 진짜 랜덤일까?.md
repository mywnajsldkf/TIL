# [C/C++] 랜덤함수에서 대해 알아보기

- c++로 랜덤함수를 마주했다. 나는 rand를 이용해 구했다. 여러번 돌려봐도 똑같은 값만 나오더라.
- rand, srand, time 랜덤함수를 공부해보자



### rand 함수 원형과 사용법

#### 1. 헤더파일

C언어 <stdlib.h> / C++ <cstlib>

#### 2. 함수 원형

`int rand(void)`

#### 3. 하는 일

- 랜덤한 숫자를 반환한다.

- 범위가 1 ~ RAND_MAX 이고 RAND_MAX는 stdlib.h 헤더파일에 저장되어있다.

  - RAND_MAX = 32767 이므로 사이의 값만 반환된다.

  ```c++
  #include<cstdlib>
  #include <iostream>
  #include<ctime>
  
  using namespace std;
  
  int main(void)
  {
      printf("rand : %d\n", rand());
      printf("rand : %d\n", rand());
      printf("rand : %d\n", rand());
  
      return 0;
  }
  ```

  <img width="554" alt="스크린샷 2020-09-23 오후 10 33 20" src="https://user-images.githubusercontent.com/47661695/94020067-762da200-fded-11ea-92f2-6673141b0734.png">

### srand 함수원형과 사용법

#### 1. 헤더파일 

c언어 <stdlib.h> / C++ <cstdlib>

#### 2. 함수원형

`void srand (unsigned int seed)`

#### 3. srand 함수가 하는 일

