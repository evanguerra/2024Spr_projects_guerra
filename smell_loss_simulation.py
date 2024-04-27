import numpy as np
import pandas as pd
import xport


class Person:
    def __init__(self, age, covid_status, location, congenital_smell_loss_prob=0.01):
        self.age = age
        self.covid_status = covid_status
        self.congenital_smell_loss_prob = congenital_smell_loss_prob
        self.smell_loss = False
        self.is_smoker = False
        self.location = location

        # Assign congenital smell loss
        if np.random.rand() < self.congenital_smell_loss_prob:
            self.smell_loss = True

    def get_infected(self, virus_type, smell_loss_prob):
        if virus_type == 'covid':
            # Define infection probability based on age
            infection_prob = calculate_covid_infection_probability(self.age)
        else:
            # Define infection probability for non-covid virus
            infection_prob = calculate_non_covid_infection_probability(self.age)

        # Simulate infection and smell loss
        if np.random.rand() < infection_prob:
            if np.random.rand() < smell_loss_prob:
                self.smell_loss = True


def calculate_covid_infection_probability(age):
    """
    Calculates the covid infection probability for a person based on age
    :param age: age of the person
    :return: covid infection probability
    """
    # TODO update this with actual calculation
    return 0.1


def calculate_non_covid_infection_probability(age):
    """
        Calculates the non-covid infection probability for a person based on age
        :param age: age of the person
        :return: non-covid infection probability
        """
    # TODO update this with actual calculation
    return 0.05


def calculate_smell_loss_probability(age):
    """
        Calculates the smell loss probability for a person based on age and smoker status
        :param age: age of the person
        :return: smell loss probability
        """
    # TODO update this with actual calculation
    return 0.2


def load_smell_loss_data(year):
    """
    Loads the smell loss data from the csv file and returns it as a pandas dataframe
    """
    if year == 2011:
        data = pd.read_csv('NHANES20112012/2011_smell_loss_data.csv')
    elif year == 2014:
        data = pd.read_csv('NHANES20132014/2014_smell_loss_data.csv')
    elif year == 2020 or year == 2021:
        data = pd.read_csv('GCCR002/data-clean.csv')
    else:
        raise ValueError("Invalid year provided")

    return data


def move_people(population):
    """
    Move the people from one location to another
    """
    for person in population:
        person.location += np.random.randint(-3, 4)


def run_simulation(population_size, num_iterations, transmission_distance=10):
    """
    Run the simulation with the given population size and number of iterations and transmission distance
    """
    population = []

    # Load smell loss data for COVID pandemic
    covid_smell_loss_data = load_smell_loss_data(2020)

    # Generate population
    for _ in range(population_size):
        age = np.random.randint(1, 101)
        # TODO update this with real initial infection population
        covid_status = np.random.choice(['infected', 'not_infected'],
                                        p=[0.1, 0.9])
        location = np.random.randint(0, 100)
        person = Person(age, covid_status, location)
        population.append(person)

    # Run simulation
    for _ in range(num_iterations):
        move_people(population)
        for i, person in enumerate(population):
            for j, other_person in enumerate(population):
                if i != j:
                    if np.abs(person.location - other_person.location) <= transmission_distance:
                        if person.covid_status == 'infected':
                            person.get_infected('covid', calculate_smell_loss_probability(person.age))
                        elif other_person.covid_status == 'infected':
                            other_person.get_infected('covid', calculate_smell_loss_probability(other_person.age))

    smell_loss_counts = sum(person.smell_loss for person in population)
    total_population = len(population)
    smell_loss_percentage = (smell_loss_counts / total_population) * 100

    return smell_loss_percentage


if __name__ == "__main__":
    pop_size = 1000
    iterations = 100
    infection_distance = 5

    smell_loss = run_simulation(pop_size, iterations, infection_distance)
    print(f"Percentage of population with smell loss: {smell_loss:.2f}%")
