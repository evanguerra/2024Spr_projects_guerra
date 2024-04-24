class Person:
    def __init__(self):
        self.age = 0
        self.infected = False
        self.COVID = False
        self.is_smoker = False
        self.smell_loss = False

    def infect(self):
        """
        Infects the person with a non-COVID virus if they get close enough to another infection person,
        at the rate of infection for flu/cold.
        Calls smell_loss at rate of smell loss due to non-covid virus
        :return: none
        """

    def infect_COVID(self):
        """
        Infects the person with COVID virus if they get close enough to another infected person,
        at the rate of COVID infection
        Calls smell_loss at rate of COVID smell loss
        :return: none
        """
    def smell_loss(self):
        """
        Updates smell loss variable to True if not already True
        :return: none
        """