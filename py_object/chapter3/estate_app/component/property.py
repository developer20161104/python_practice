class Property:
    def __init__(self, square_feet='', beds='', baths='', **kwargs):
        super().__init__(**kwargs)
        self.square_feet = square_feet
        self.beds = beds
        self.baths = baths

    def display(self):
        print("property details:")
        print("square footage: {}".format(self.square_feet))
        print("bedroom: {}".format(self.beds))
        print("bathroom: {}".format(self.baths))

    # 静态方法只与类有关，因此没有self参数
    def prompt_init():
        return dict(square_feet=input("input square feet: "),
                    beds=input("input bedroom: "),
                    baths=input("input bathroom: "))

    prompt_init = staticmethod(prompt_init)


class Apartment(Property):
    valid_laundries = ("coin", "ensuite", "none")
    valid_balconies = ("yes", "no", "solarium")

    def __init__(self, balcony='', laundry='', **kwargs):
        super().__init__()
        self.balcony = balcony
        self.laundry = laundry

    def display(self):
        super().display()
        print("apartment details: ")
        print("laundries: {}".format(self.laundry))
        print("has balcony: {}".format(self.balcony))

    def prompt_init():
        parent_init = Property.prompt_init()

        laundry = ''
        while laundry.lower() not in Apartment.valid_laundries:
            laundry = input("what laundry facilities does the property have? ({})".format(
                "、".join(Apartment.valid_laundries)
            ))

        balcony = ''
        while balcony.lower() not in Apartment.valid_balconies:
            balcony = input("does the property have a balcony? ({})".format(
                "、".join(Apartment.valid_balconies)
            ))

        parent_init.update({
            "laundry": laundry,
            "balcony": balcony
        })

        return parent_init

    prompt_init = staticmethod(prompt_init)