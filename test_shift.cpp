#include <iostream>

int main() {
    int32_t x = 0x80000000;
    std::cout << x << std::endl;
    x >>= 31;
    std::cout << x << std::endl;
}