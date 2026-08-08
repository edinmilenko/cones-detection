#include "labeler.hpp"
#include "dataloader.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    
    try {
        //  LEAVE DISABLED for first run only
        //  altrimenti rifa ogni volta, o cambiamo con un check oppure documentiamo dove mettere i tre zip prima della prima run
        //  valutare check hardcodato?
        //dataLoader();

        // labeler sarà da togliere dalla consegna, credo sia utile solo per fare bene i punti 2
        labeler();

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}