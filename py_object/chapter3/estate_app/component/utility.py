from component.property import Property, House


class Purchase:
    def __init__(self, price='', taxes='', **kwargs):
        super().__init__(**kwargs)
        self.price = price
        self.taxes = taxes

    def display(self):
        super().display()
        print("purchase details")
        print("selling prices: {}".format(self.price))
        print("estimated taxes: {}".format(self.taxes))

    def prompt_init():
        return dict(
            price=input("selling prices?"),
            taxes=input("estimated taxes?")
        )

    prompt_init = staticmethod(prompt_init)


class Rental:
    def __init__(self, furnished='', utilities='', rent='', **kwargs):
        super().__init__(**kwargs)
        self.furnished = furnished
        self.utilities = utilities
        self.rent = rent

    def display(self):
        super().display()
        print("Rental details")
        print("rent: {}".format(self.rent))
        print("estimated utilities: {}".format(self.utilities))
        print("furnished: {}".format(self.furnished))

    def prompt_init():
        return dict(
            rent=input("input rent:"),
            utilities=input("utilities:"),
            furnished=Property.get_valid_input("is furnished?", ("yes", "no"))
        )

    prompt_init = staticmethod(prompt_init)


class HouseRental(House, Rental):
    def prompt_init():
        init = House.prompt_init()
        init.update(Rental.prompt_init())
        return init

    prompt_init = staticmethod(prompt_init)
