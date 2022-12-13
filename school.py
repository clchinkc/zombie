"""
In a school, human dynamics can be modeled as follows:

1. Students form social groups and friendships based on shared interests, backgrounds, and personalities. These groups often have a hierarchical structure, with some individuals holding more influence and popularity than others.

2. Teachers and administrators play a central role in shaping the social dynamic of the school, providing guidance, enforcing rules, and creating a sense of community.

3. Peer pressure, competition, and social norms can influence students' behavior and decision-making, leading to conformity, conformity to deviance, or nonconformity.

4. Conflicts and disagreements can arise within and between social groups, and may be resolved through negotiation, mediation, or intervention from teachers or administrators.

5. Over time, the social dynamic of a school may shift and evolve, as students graduate, new students join the school, and external factors such as changing cultural norms or technology affect the social landscape.
"""

"""

Second, the code could be made more efficient by using data structures such as dictionaries and sets. For example, the students and teachers lists could be converted to sets to allow for faster membership checking and set operations such as union and intersection. The friendships list could be converted to a set of tuples, or to a dictionary where the keys are the student names and the values are sets of the names of their friends. The hierarchy dictionary could be inverted to create a dictionary where the keys are the student names and the values are the social groups they belong to. This would make it easier to quickly look up a student's friends, social groups, or peer pressure level.


"""
"""
Here are a few suggestions:

The current code only prints the output of each function, but it does not save the results in any variable. As a result, the output of one function cannot be used as input for another function. This can be improved by returning the results from each function and storing them in variables for further use.

Make functions to generate the input parameters of the School class.
"""

class School:
    def __init__(self, students, teachers, social_groups, friendships, hierarchy, peer_pressure, social_norms):
        self.students = students
        self.teachers = teachers
        self.social_groups = social_groups
        self.friendships = friendships
        self.hierarchy = hierarchy
        self.peer_pressure = peer_pressure
        self.social_norms = social_norms

    def form_friendships(self):
        """Prints a message for each friendship that is formed."""
        for friendship in self.friendships:
            print(f"{friendship[0]} and {friendship[1]} have formed a friendship.")

    def assign_to_social_groups(self):
        """Prints the members of each social group based on the given hierarchy."""
        for group in self.social_groups:
            members = self.hierarchy[group]
            print(f"{group}: {members}")

    def enforce_rules(self):
        """Prints a message for each student indicating the social norms they are expected to follow."""
        for student in self.students:
            for norm in self.social_norms:
                print(f"{student} is expected to {norm}.")

    def resolve_conflicts(self):
        """Prints a message for each student indicating possible conflicts and suggesting a teacher for resolution."""
        for student in self.students:
            for group in self.social_groups:
                if student in self.hierarchy[group]:
                    for other_student in self.hierarchy[group]:
                        if student != other_student:
                            print(f"{student} and {other_student} are in the same social group and may have conflicts.")
                            print(f"A teacher, such as {self.teachers[0]}, can help resolve these conflicts.")

    def evolve_social_dynamic(self):
        """Updates the list of students and the hierarchy of social groups to reflect changes over time."""
        new_students = ["Emily", "Daniel"]
        graduating_students = ["John", "Sara"]
        self.students = [student for student in self.students if student not in graduating_students]
        self.students.extend(new_students)
        print(f"New students: {new_students}")
        print(f"Graduating students: {graduating_students}")
        print(f"Updated list of students: {self.students}")

        updated_hierarchy = {"athletic club": ["Emily", "Tom"], "science club": ["Daniel"], "chess club": ["Jane"]}
        self.hierarchy = updated_hierarchy
        print(f"Updated hierarchy of social groups: {self.hierarchy}")

    def apply_peer_pressure(self):
        """Prints a message for each student indicating their level of peer pressure."""
        for student in self.students:
            pressure = self.peer_pressure[student]
            print(f"{student} is experiencing {pressure} peer pressure.")

    def provide_guidance(self):
        for student in self.students:
            for teacher in self.teachers:
                print(f"{teacher} is providing guidance to {student}.")
                
    def add_student(self, name, social_groups, peer_pressure):
        """Adds a student to the school with the given name, social groups, and peer pressure level."""
        self.students.append(name)
        self.hierarchy.update(social_groups)
        self.peer_pressure[name] = peer_pressure

    def remove_student(self, name):
        """Removes a student from the school with the given name."""
        self.students.remove(name)
        self.peer_pressure.pop(name, None)
        for members in self.hierarchy.items():
            if name in members:
                members.remove(name)
                self.hierarchy.update(members)

def add_teacher(self, teacher):
    """Adds a new teacher to the school."""
    self.teachers.append(teacher)
    print(f"{teacher} has been added to the school.")

def remove_teacher(self, teacher):
    """Removes a teacher from the school."""
    self.teachers.remove(teacher)
    print(f"{teacher} has been removed from the school.")

def add_social_group(self, group, members):
    """Adds a new social group to the school, with the given members."""
    self.social_groups.append(group)
    self.hierarchy[group] = members
    print(f"{group} has been added to the school, with members: {members}.")

def remove_social_group(self, group):
    """Removes a social group from the school."""
    self.social_groups.remove(group)
    del self.hierarchy[group]
    print(f"{group} has been removed from the school.")

def update_peer_pressure(self, student, pressure):
    """Updates a student's peer pressure level."""
    self.peer_pressure[student] = pressure
    print(f"{student}'s peer pressure level has been updated to {pressure}.")

def update_social_groups(self, student, groups):
    """Updates a student's social groups."""
    for group in self.social_groups:
        if student in self.hierarchy[group]:
            self.hierarchy[group].remove(student)
    for group in groups:
        self.hierarchy[group].append(student)
    print(f"{student}'s social groups have been updated to: {groups}.")



                
students = ["John", "Sara", "Jane", "Tom"]
teachers = ["Mrs. Smith", "Mr. Johnson"]
social_groups = ["athletic club", "science club", "chess club"]
friendships = [("John", "Sara"), ("Jane", "Tom")]
hierarchy = {"athletic club": ["John", "Sara"], "science club": ["Jane"], "chess club": ["Tom"]}
peer_pressure = {"John": "high", "Sara": "low", "Jane": "medium", "Tom": "low"}
social_norms = ["respect teachers", "no cheating on tests", "no bullying"]

school = School(students, teachers, social_groups, friendships, hierarchy, peer_pressure, social_norms)

school.form_friendships()
school.assign_to_social_groups()
school.enforce_rules()
school.resolve_conflicts()
school.evolve_social_dynamic()
school.apply_peer_pressure()
school.provide_guidance()

school.add_student("Emily", {"science club": ["Emily"]}, "low")
school.remove_student("Jane")
