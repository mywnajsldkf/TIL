# [C/C++] switch문에서 문자열 사용하기

- c++ 로 계산기를 만들면서 자연스럽게 switch 문을 사용하자 에러를 마주하였다.
- 검색해보니 c++에서는 switch문 표현식에 문자열 타입을 허용하지 않고 정수 타입만 허용하고 있다고 한다. 
- 우선 for 문으로 해결하였고, 궁금해서 한번 c/c++에서도 switch문에서 문자열을 사용할 수 있는 방법을 알아보기 시작했다.



## 해시 함수

해시함수는 데이터의 효율적 관리를 위해 임의의 길이의 데이터를 고정된 길이의 데이터로 매핑하는 함수를 말한다. 

- (내가 본 글에서는) 문자열 데이터를 정수형 데이처로 매핑하는 해싱 함수를 만들어서 switch 문의 표현식으로 사용할 것을 제안하였다.

```c++
unsigned int HashCode(const char *str)
{
    unsigned int hash = 0;
    while (*str)
    {
        hash = 65599 * hash + str[0];
        str++;
    }
    return hash^(hash >> 16);
}

void find(const char* name)
{
    switch (HashCode(name))
    {
    case HashCode("Judy"):
        /* code */
        break;

    case HashCode("Sonia"):
        break;
        
    case HashCode("Nina"):
        break;
    }
}
```

빨간 줄이 쫙쫙 그어지는데 확인해보면 `식에 상수 값이 있어야 합니다. -- constexpr이 아닌 함수 "HashCode" (선언됨 줄 1)을(를) 호출할 수 없습니다.C/C++(28)` 이런 오류를 마주하게 된다.



## constexpr

C++11에 새로 추가된 키워드로 변수 또는 함수의 값을 컴파일 시점에 도출하여 상수화 시켜준다.

**C++에는 두 가지 다른 종류의 상수가 있다.**

1. **런타임 상수**(runtime constant)는 초깃값을 런타임에서만 확인할 수 있는 상수이다.
   - 컴파일러가 컴파일 시 값을 결정할 수 없다면 런타임 상수이다.
2. **컴파일 시간 상수**(compile-time constant)는 컴파일 시간에 초깃값을 확인할 수 있는 상수이다.

C++에서 고정 크기 배열의 길이를 정의하는 상황과 같은 런타임 상수 대신에 컴파일 타임 상수를 요구하는 경우가 존재한다. 이러한 특수성을 제공하기 위해 `constexper`을 도입한다.

- HashCode 함수를 `constexpr` 함수로 만들어 컴파일 시점에 상수화시켜 case 표현식에 사용하자.

  

```c++
constexpr unsigned int HashCode(const char* str)
{
    return str[0]?static_cast<unsigned int>(str[0]) + 0xEDB8832Full * HashCode(str + 1) : 8603;
}
```

- 알고리즘은 더 찾아보지 않았다. (아직 이해를 못함...!)

- `constexpr` 함수이기 때문에 파라미터에 상수 식이 전달될 경우 컴파일 시점에 값을 도축하여 상수화된다.
- 컴파일 시점에 `HashCode("Judy")`는 2460245172와 같은 형태로 치환될 것이다.



## 주의 할 점

1. 문자열 길이 제한

   재귀 호출의 깊이가 지정된 횟수를 초과할 경우 컴파일 시점에 값을 계산할 수 없다.

2. 해시 충돌

   충돌륭 0%의 완벽한 해시 함수는 없다.



**Reference)**

[][C++] [switch문에서 문자열 사용하기](https://m.blog.naver.com/PostView.nhn?blogId=devmachine&logNo=220952781191&proxyReferer=https:%2F%2Fwww.google.com%2F)