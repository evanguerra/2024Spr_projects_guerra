import numpy as np
import pandas as pd


class Person:
    """
    A class to represent a person in the smell loss simulation
    Inspired by Mr.Weible's Player class from
    https://github.com/iSchool-597PR/2024Spr_examples/blob/main/unit_08/MC_rock_paper_scissors.py
    """
    person_count = 0
    all_persons = []

    def __init__(self, age, location, infected_status='none'):
        """
        Initialize the person class with an age and a location and infected status
        :param age: The age of the person
        :param location: The starting location of the person
        :param infected_status: The status of the infected person (either 'none', 'covid', or 'flu_cold'
        """
        Person.person_count += 1
        Person.all_persons.append(self)
        self.location = location
        self.age = age
        self.infected_status = infected_status
        self.congenital_smell_loss_prob = 0.01
        self.smell_loss = False
        self.is_smoker = False

        if np.random.rand() < self.congenital_smell_loss_prob:
            self.smell_loss = True

    def get_infected(self, virus_type, smell_loss_prob, infection_prob):
        """
        Update the infected status of the person based on the infection rate and smell loss rate
        :param virus_type: The virus type of the person (either 'none', 'covid', or 'flu_cold'
        :param smell_loss_prob: The smell loss rate of the population
        :param infection_prob: The infection rate of the population
        :return: None
        """
        if np.random.rand() < infection_prob:
            self.infected_status = virus_type
            if np.random.rand() < smell_loss_prob:
                self.smell_loss = True

    def reset_stats(self):
        """From Mr.Weible's Player class from
        https://github.com/iSchool-597PR/2024Spr_examples/blob/main/unit_08/MC_rock_paper_scissors.py
        """
        self.infected_status = 'none'
        self.congenital_smell_loss_prob = 0.01
        self.smell_loss = False
        self.is_smoker = False

        if np.random.rand() < self.congenital_smell_loss_prob:
            self.smell_loss = True

    @staticmethod
    def reset_all_stats():
        """Resets all 'memory' or other counters in ALL Players, to prepare for
        a new tournament unaffected by previous matches."""
        for p in Person.all_persons:
            p.reset_stats()


def calculate_covid_infection_probability(year):
    """
    Calculates the covid infection probability for a person
    :param year: The year for the simulation
    :return: covid infection probability
    """
    cases = load_case_data(year)

    covid_prob = (cases / 333271411)*100  # 2020-2022 estimated US population from census.gov

    return covid_prob


def calculate_non_covid_infection_probability(year):
    """
        Calculates the non-covid infection probability for a person
        :param year: The year for the simulation
        :return: non-covid infection probability
        """
    data = load_smell_loss_data(year)

    non_covid_prob = (data['CSQ020'] == 1).mean()

    return non_covid_prob


def calculate_smell_loss_probability(year):
    """
        Calculates the smell loss probability for a person based on the smell loss rate of the virus type
        :param year: The year for the simulation
        :return: smell loss probability
        """
    data = load_smell_loss_data(year)

    if year == 2021:
        cases = load_case_data(year)

        smell_loss_data = data['Symptoms_changes_in_smell']
        smell_loss_count = smell_loss_data.sum()
        smell_loss_prob = (smell_loss_count / cases)*100
    else:
        smell_loss_prob = (data['CSQ010'] == 1).mean()

    return smell_loss_prob


def load_smell_loss_data(year):
    """
    Loads the smell loss data from the csv file and returns it as a pandas dataframe
    :param year: The year for the simulation
    :return: the dataframe
    """
    if year == 2011:
        data = pd.read_csv('NHANES20112012/2011_smell_loss_data.csv')
    elif year == 2014:
        data = pd.read_csv('NHANES20132014/2014_smell_loss_data.csv')
    elif year == 2021:
        data = pd.read_csv('GCCR002/data-clean.csv')
    else:
        raise ValueError("Invalid year provided")

    return data


def load_case_data(year):
    """
    Loads the covid case data from the csv file and returns it as a pandas dataframe
    :param year: The year for the simulation
    :return: the dataframe
    """
    data = pd.read_csv('WHO-COVID-19-global-data.csv')
    data['Date_reported'] = pd.to_datetime(data['Date_reported'])
    filtered_data = data[(data['Country'] == 'United States of America') & (data['Date_reported'].dt.year <= year)]
    cases = filtered_data['New_cases'].sum()

    return cases


def move_people(population):
    """
    Move the people from one location to another
    :param population: The population of the simulation
    """
    for person in population:
        person.location += np.random.randint(-3, 4)


def run_simulation(population_size, num_iterations, transmission_distance=10, year=2011, initial_infected=10):
    """
    Run the simulation with the given population size and number of iterations and transmission distance
    :param population_size: The population size of the simulation
    :param num_iterations: The number of iterations of the simulation
    :param transmission_distance: The transmission distance of the simulation
    :param year: The year of the simulation
    :param initial_infected: The initial infected persons in the simulation
    :return: None
    >>> np.random.seed(42)
    >>> aggregate_stats, population = run_simulation(100, 10, transmission_distance=5, year=2011, initial_infected=10)  # doctest: +ELLIPSIS
    Iteration   0: Smell Loss Percentage: ...
    Iteration   1: Smell Loss Percentage: ...
    Iteration   2: Smell Loss Percentage: ...
    Iteration   3: Smell Loss Percentage: ...
    Iteration   4: Smell Loss Percentage: ...
    Iteration   5: Smell Loss Percentage: ...
    Iteration   6: Smell Loss Percentage: ...
    Iteration   7: Smell Loss Percentage: ...
    Iteration   8: Smell Loss Percentage: ...
    Iteration   9: Smell Loss Percentage: ...
    >>> len(aggregate_stats), len(population)
    (10, 100)
    >>> np.random.seed(42)
    >>> aggregate_stats, population = run_simulation(100, 10, transmission_distance=5, year=2021, initial_infected=10)  # doctest: +ELLIPSIS
    Iteration   0: Smell Loss Percentage: ...
    Iteration   1: Smell Loss Percentage: ...
    Iteration   2: Smell Loss Percentage: ...
    Iteration   3: Smell Loss Percentage: ...
    Iteration   4: Smell Loss Percentage: ...
    Iteration   5: Smell Loss Percentage: ...
    Iteration   6: Smell Loss Percentage: ...
    Iteration   7: Smell Loss Percentage: ...
    Iteration   8: Smell Loss Percentage: ...
    Iteration   9: Smell Loss Percentage: ...
    >>> len(aggregate_stats), len(population)
    (10, 100)
    >>> np.random.seed(42)
    >>> aggregate_stats, population = run_simulation(100, 10, transmission_distance=5, year=2019, initial_infected=10)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Invalid year provided
    """
    population = []
    aggregate_stats = []

    if year == 2021:
        infection_prob = calculate_covid_infection_probability(year)
        virus = 'covid'
    else:
        # Define infection probability for non-covid virus
        infection_prob = calculate_non_covid_infection_probability(year)
        virus = 'flu_cold'

    smell_loss_probability = calculate_smell_loss_probability(year)

    for _ in range(population_size):
        age = np.random.randint(1, 101)
        infected_status = virus if initial_infected > 0 else 'none'
        initial_infected -= 1
        locale = np.random.randint(0, 100)
        person = Person(age, locale, infected_status)
        population.append(person)

    for _ in range(num_iterations):
        move_people(population)

        for i, person in enumerate(population):
            if person.infected_status == 'none':
                for j, other_person in enumerate(population):
                    if i != j:
                        if np.abs(person.location - other_person.location) <= transmission_distance:
                            if other_person.infected_status != 'none':
                                person.get_infected(virus, smell_loss_probability, infection_prob)

        smell_loss_count = sum(person.smell_loss for person in population)
        total_population = len(population)
        smell_loss_percentage = (smell_loss_count / total_population) * 100
        print(f"Iteration {_:>3}: Smell Loss Percentage: {smell_loss_percentage:.2f}%")
        aggregate_stats.append(smell_loss_percentage)

    return aggregate_stats, population


if __name__ == "__main__":
    pop_size = 1000
    iterations = 100
    infection_distance = 5

    run_simulation(pop_size, iterations, infection_distance, year=2011)

    run_simulation(pop_size, iterations, infection_distance, year=2014)

    run_simulation(pop_size, iterations, infection_distance, year=2021)
