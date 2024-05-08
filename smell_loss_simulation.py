import numpy as np
import pandas as pd


class Person:
    """
    A class to represent a person in the smell loss simulation
    Inspired by Mr. Weible's Player class from
    https://github.com/iSchool-597PR/2024Spr_examples/blob/main/unit_08/MC_rock_paper_scissors.py
    """
    person_count = 0
    all_persons = []

    def __init__(self, age, location, infected_status='none'):
        """
        Initialize the Person class with age, location, and infected status
        :param age: The age of the person
        :param location: The starting location of the person
        :param infected_status: The status of the infected person
            (either 'none', 'covid', or 'flu_cold')
        """
        Person.person_count += 1
        Person.all_persons.append(self)
        self.location = location
        self.age = age
        self.infected_status = infected_status
        self.congenital_smell_loss_prob = 0.01
        self.smell_loss = False
        self.taste_loss = False
        self.flavor_loss = False
        self.is_smoker = False
        self.poverty = False

        if np.random.rand() < self.congenital_smell_loss_prob:
            self.smell_loss = True

    def get_infected(self, virus_type, smell_loss_prob, infection_prob, taste_loss_prob, flavor_loss_prob):
        """
        Update the infected status of the person based on the infection rate and smell loss rate
        :param virus_type: The virus type of the person
            (either 'none', 'covid', or 'flu_cold')
        :param smell_loss_prob: The smell loss rate of the population
        :param infection_prob: The infection rate of the population
        :param taste_loss_prob: The taste loss rate of the population
        :param flavor_loss_prob: The flavor loss rate of the population
        :return: None
        """

        if np.random.rand() < infection_prob:
            self.infected_status = virus_type
            self.smell_loss = True if np.random.rand() < smell_loss_prob else False
            self.taste_loss = True if np.random.rand() < taste_loss_prob else False
            self.flavor_loss = True if np.random.rand() < flavor_loss_prob else False

    def reset_stats(self):
        """Reset the statistics of a person"""
        self.infected_status = 'none'
        self.congenital_smell_loss_prob = 0.01
        self.smell_loss = False
        self.is_smoker = False

        if np.random.rand() < self.congenital_smell_loss_prob:
            self.smell_loss = True

    @staticmethod
    def reset_all_stats():
        """Reset statistics for all persons"""
        for p in Person.all_persons:
            p.reset_stats()


def calculate_smoking_probability(year):
    """
    Calculate the smoking probability for a given year
    :param year: The year for the simulation
    :return: Smoking probability
    """
    if year != 2021:
        data = load_smoking_data(year)
        smoking_data = data[
            data['SMQ621'].isin([3, 4, 5, 6, 7, 8]) & ~data['SMQ621'].isin([1, 2, 77, 99, float('nan')])]
        smoking_count = len(smoking_data)
        total_count = len(data)
        smoking_prob = (smoking_count / total_count) * 100
    else:
        data = load_smell_loss_data(year)
        smoking_prob = (data['Combustible_cigarette_frequency'] <= 1).mean()

    return smoking_prob


def start_smoking(population, year):
    """
    Set a person's is_smoker attribute to True for a percentage of the population
    :param population: The population of the simulation
    :param year: The year of the simulation
    :return: None
    """
    smoking_prob = calculate_smoking_probability(year)
    num_smokers = int(len(population) * (smoking_prob / 100))

    for person in population[:num_smokers]:
        person.is_smoker = True


def calculate_poverty_probability(year):
    """
    Calculate the poverty probability for a given year
    :param year: The year for the simulation
    :return: Poverty probability
    """
    data = load_income_data(year)
    total_count = len(data)
    poverty_data = data[data['INDFMPIR'] == 5]
    poverty_count = len(poverty_data)
    poverty_prob = (poverty_count / total_count) * 100

    return poverty_prob


def get_income(population, year):
    """
    Set a person's poverty attribute to True for a percentage of the population
    :param population: The population of the simulation
    :param year: The year of the simulation
    :return: None
    """
    poverty_prob = calculate_poverty_probability(year)
    num_poverty = int(len(population) * (poverty_prob / 100))

    for person in population[:num_poverty]:
        person.poverty = True


def calculate_covid_infection_probability(year):
    """
    Calculate the COVID-19 infection probability for a given year
    :param year: The year for the simulation
    :return: COVID-19 infection probability
    """
    cases, _ = load_case_data(year)

    covid_prob = (cases / 333271411) * 100  # 2020-2022 estimated US population from census.gov

    return covid_prob


def calculate_non_covid_infection_probability(year):
    """
    Calculate the non-COVID-19 infection probability for a given year
    :param year: The year for the simulation
    :return: Non-COVID-19 infection probability
    """
    data = load_smell_loss_data(year)

    non_covid_prob = (data['CSQ200'] == 1).mean()

    return non_covid_prob


def calculate_smell_loss_probability(year):
    """
    Calculate the smell loss probability for a given year
    :param year: The year for the simulation
    :return: Smell loss probability
    """
    data = load_smell_loss_data(year)

    if year == 2021:
        cases, _ = load_case_data(year)

        smell_loss_data = data['Symptoms_changes_in_smell']
        smell_loss_count = smell_loss_data.sum()
        smell_loss_prob = (smell_loss_count / cases) * 100
    else:
        smell_loss_prob = (data['CSQ010'] == 1).mean()

    return smell_loss_prob


def calculate_taste_loss_probability(year):
    """
    Calculate the taste loss probability for a given year
    :param year: The year for the simulation
    :return: Taste loss probability
    """
    data = load_smell_loss_data(year)

    if year == 2021:
        cases, _ = load_case_data(year)

        columns_to_check = ['Changes_in_basic_tastes_sweet', 'Changes_in_basic_tastes_salty',
                            'Changes_in_basic_tastes_sour', 'Changes_in_basic_tastes_bitter',
                            'Changes_in_basic_tastes_savory/umami']
        data['taste_loss'] = data[columns_to_check].any(axis=1).astype(int)
        taste_loss_data = data['taste_loss']
        taste_loss_count = taste_loss_data.sum()
        taste_loss_prob = (taste_loss_count / cases) * 100
    else:
        taste_loss_prob = (data['CSQ080'] == 1).mean()

    return taste_loss_prob


def calculate_flavor_loss_probability(year):
    """
    Calculate the flavor loss probability for a given year
    :param year: The year for the simulation
    :return: Flavor loss probability
    """
    data = load_smell_loss_data(year)

    if year == 2021:
        cases, _ = load_case_data(year)

        flavor_loss_data = data['Symptoms_changes_in_food_flavor']
        flavor_loss_count = flavor_loss_data.sum()
        flavor_loss_prob = (flavor_loss_count / cases) * 100
    else:
        flavor_loss_prob = (data['CSQ100'] == 1).mean()

    return flavor_loss_prob


def calculate_get_better_probability(year):
    """
    Calculate the probability of getting better for a given year
    :param year: The year for the simulation
    :return: Get better probability
    """
    cases, deaths = load_case_data(year)

    if year == 2021:
        get_better_prob = 100 - ((deaths / cases) * 100)
    elif year == 2011:
        # Data not available in downloadable format from https://archive.cdc.gov/
        get_better_prob = (12447 / 9315621) * 100
    else:
        # Data not available in downloadable format from https://archive.cdc.gov/
        get_better_prob = (37930 / 29739994)

    return get_better_prob


def load_smell_loss_data(year):
    """
    Load smell loss data from the CSV file and return it as a pandas DataFrame
    :param year: The year for the simulation
    :return: The DataFrame containing smell loss data
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


def load_income_data(year):
    """
    Load income data from the CSV file and return it as a pandas DataFrame
    :param year: The year for the simulation
    :return: The DataFrame containing income data
    """
    if year == 2011:
        data = pd.read_csv('NHANES20112012/2011_income_data.csv')
    elif year == 2014:
        data = pd.read_csv('NHANES20132014/2014_income_data.csv')
    else:
        raise ValueError("Invalid year provided")

    return data


def load_smoking_data(year):
    """
    Load smoking data from the CSV file and return it as a pandas DataFrame
    :param year: The year for the simulation
    :return: The DataFrame containing smoking data
    """
    if year == 2011:
        data = pd.read_csv('NHANES20112012/smoking_data.csv')
    elif year == 2014:
        data = pd.read_csv('NHANES20132014/2013smoking_data.csv')
    else:
        raise ValueError("Invalid year provided")

    return data


def load_case_data(year):
    """
    Load COVID-19 case data from the CSV file and return the total cases and deaths
    :param year: The year for the simulation
    :return: Total cases and deaths
    """
    data = pd.read_csv('WHO-COVID-19-global-data.csv')
    data['Date_reported'] = pd.to_datetime(data['Date_reported'])
    filtered_data = data[(data['Country'] == 'United States of America') & (data['Date_reported'].dt.year <= year)]
    cases = filtered_data['New_cases'].sum()
    deaths = filtered_data['New_deaths'].sum()

    return cases, deaths


def move_people(population):
    """
    Move people from one location to another
    :param population: The population of the simulation
    """
    for person in population:
        person.location += np.random.randint(-3, 4)


def run_simulation(population_size, num_iterations, transmission_distance=10, year=2011, initial_infected=10):
    """
    Run the simulation with the given parameters
    :param population_size: The population size of the simulation
    :param num_iterations: The number of iterations of the simulation
    :param transmission_distance: The transmission distance of the simulation
    :param year: The year of the simulation
    :param initial_infected: The initial infected persons in the simulation
    :return: Aggregate statistics and the ending population
    >>> np.random.seed(42)
    >>> aggregate_statistics, pop = run_simulation(100, 10, transmission_distance=5,
    ... year=2011, initial_infected=10)  # doctest: +ELLIPSIS
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
    >>> len(aggregate_statistics), len(pop) # doctest: +ELLIPSIS
    (10, ...)
    >>> np.random.seed(42)
    >>> aggregate_statistics, pop = run_simulation(100, 10, transmission_distance=5,
    ... year=2021, initial_infected=10)  # doctest: +ELLIPSIS
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
    >>> len(aggregate_statistics), len(pop) # doctest: +ELLIPSIS
    (10, ...)
    >>> np.random.seed(42)
    >>> aggregate_statistics, pop = run_simulation(100, 10, transmission_distance=5,
    ... year=2019, initial_infected=10)  # doctest: +ELLIPSIS
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
        infection_prob = calculate_non_covid_infection_probability(year)
        virus = 'flu_cold'

    smell_loss_probability = calculate_smell_loss_probability(year)
    taste_loss_probability = calculate_taste_loss_probability(year)
    flavor_loss_probability = calculate_flavor_loss_probability(year)
    get_better_probability = calculate_get_better_probability(year)

    for _ in range(population_size):
        age = np.random.randint(1, 101)
        infected_status = virus if initial_infected > 0 else 'none'
        initial_infected -= 1
        locale = np.random.randint(0, 100)
        person = Person(age, locale, infected_status)
        population.append(person)

    for _ in range(num_iterations):
        start_smoking(population, year)
        move_people(population)
        if year != 2021:
            get_income(population, year)

        for i, person in enumerate(population):
            if person.infected_status != 'none':
                if np.random.rand() < get_better_probability:
                    person.infected_status = 'none'
                    person.smell_loss = False
                    person.taste_loss = False
                    person.flavor_loss = False
                else:
                    population.pop(i)

        for i, person in enumerate(population):
            if person.infected_status == 'none':
                for j, other_person in enumerate(population):
                    if i != j:
                        if np.abs(person.location - other_person.location) <= transmission_distance:
                            if other_person.infected_status != 'none':
                                person.get_infected(virus, smell_loss_probability, infection_prob,
                                                    taste_loss_probability, flavor_loss_probability)

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
