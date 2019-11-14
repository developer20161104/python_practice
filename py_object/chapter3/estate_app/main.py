from component import property, utility

if __name__ == '__main__':
    init = utility.HouseRental.prompt_init()
    house = utility.HouseRental(**init)
    house.display()