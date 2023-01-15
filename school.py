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

"""
How can you make use of this in a zombie apocalypse simulation?

In a zombie apocalypse simulation, the human dynamics modeled in the School class could be applied to a group of survivors. The survivors could form social groups and friendships based on shared interests, backgrounds, and personalities. These groups would likely have a hierarchical structure, with some individuals holding more influence and popularity than others. The leader(s) of the group would play a central role in shaping the social dynamic, providing guidance, enforcing rules, and creating a sense of community.

Peer pressure, competition, and social norms would also influence the survivors' behavior and decision-making, leading to conformity, conformity to deviance, or nonconformity. Conflicts and disagreements would arise within and between social groups and would need to be resolved through negotiation, mediation, or intervention from the leader(s). As the group encounters new survivors, loses members or faces different challenges, the social dynamic of the group would shift and evolve.

It could also be interesting to incorporate the concept of "peer pressure" to the zombies, where some zombies would be more aggressive and leader-like, while others would be more docile, and how this would affect the group dynamic of the zombies.

Overall, the School class can be used as a starting point for modeling human dynamics in a zombies apocalypse simulation and can be further developed and adapted to suit the specific needs of the simulation.
"""

class Student:
    def __init__(self, name, group=None, pressure=0):
        self.name = name
        self.group = group
        self.pressure = pressure
        self.friends = []

    def add_friend(self, student):
        if student not in self.friends:
            self.friends.append(student)
            student.friends.append(self)

    def remove_friend(self, student):
        if student in self.friends:
            self.friends.remove(student)
            student.friends.remove(self)

    def __str__(self):
        return self.name

class School:
    def __init__(self, students, teachers, social_groups, hierarchy, peer_pressure, social_norms):
        self.teachers = teachers
        self.students = {student.name: student for student in students}
        self.social_groups = social_groups
        self.hierarchy = hierarchy
        self.peer_pressure = peer_pressure
        self.social_norms = social_norms

    def add_student(self, student: Student):
        """Adds a student to the school"""
        self.students[student.name] = student
        self.hierarchy[student.group].append(student)
        self.peer_pressure[student.name] = student.pressure
        print(f"{student} has been added to the school.")

    def remove_student(self, student):
        """Removes a student from the school with the given name."""
        student_obj = self.students.get(student)
        if student_obj is None:
            print(f"{student} not found in school.")
            return
        del self.students[student]
        self.hierarchy[student_obj.group].remove(student_obj)
        del self.peer_pressure[student]
        for friend in student_obj.friends:
            friend.friends.remove(student_obj)
        print(f"{student} has been removed from the school.")

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
        student_obj = self.students.get(student)
        if student_obj is None:
            print(f"{student} not found in school.")
            return
        student_obj.peer_pressure = pressure
        self.peer_pressure[student] = pressure
        print(f"{student}'s peer pressure level has been updated to {pressure}.")

    def update_social_groups(self, student, groups):
        """Updates a student's social groups."""
        student_obj = self.students.get(student)
        if student_obj is None:
            print(f"{student} not found in school.")
            return
        old_group = student_obj.group
        self.hierarchy[old_group].remove(student_obj)
        student_obj.group = groups
        self.hierarchy[groups].append(student_obj)
        print(f"{student}'s social groups have been updated to: {groups}.")

    def form_friendship(self, student1, student2):
        student1 = self.students.get(student1)
        student2 = self.students.get(student2)
        if student1 is None or student2 is None:
            print(f"One or both of the students {student1} and {student2} not found.")
            return
        student1.add_friend(student2)
        print(f"{student1} and {student2} have formed a friendship.")

    def assign_to_social_groups(self):
        """Prints the members of each social group based on the given hierarchy."""
        for group in self.social_groups:
            members = self.hierarchy.get(group)
            if members is None:
                print(f"{group} not found.")
                continue
            print(f"{group}: {members}")
            
    def get_student_group(self, student):
        student = self.students.get(student)
        if student is None:
            print(f"{student} not found.")
            return None
        group = student.group
        if group is None:
            print(f"{student} not found in any group.")
            return None
        if group not in self.hierarchy:
            print(f"{group} not found.")
            return None
        return group

    def evolve_social_dynamic(self, new_students, new_hierarchy, new_friendships):
        """Updates the list of students and the hierarchy of social groups to reflect changes over time."""

        # Add new students to the list of students
        for student in new_students:
            self.add_student(student)

        # Add new social groups to the hierarchy
        for group, students in new_hierarchy.items():
            self.add_social_group(group, students)

        # Add new friendships
        for friendship in new_friendships:
            self.form_friendship(friendship[0], friendship[1])

    def enforce_rules(self):
        """Prints a message for each student indicating the social norms they are expected to follow."""
        for student in self.students.values():
            for norm in self.social_norms:
                print(f"{student} is expected to {norm}.")

    def resolve_conflicts(self):
        """Prints a message for each student indicating possible conflicts and suggesting a teacher for resolution."""
        for student in self.students.values():
            group = student.group
            if group is None:
                print(f"{student} not found in any group.")
                continue
            if group not in self.hierarchy:
                print(f"{group} not found.")
                continue
            for other_student in self.hierarchy[group]:
                if student != other_student and other_student in student.friends:
                    print(f"{student} and {other_student} are in the same social group and may have conflicts.")
                    print(f"A teacher, such as {self.teachers[0]}, can help resolve these conflicts.")

    def apply_peer_pressure(self):
        """Prints a message for each student indicating their level of peer pressure."""
        for student in self.students:
            pressure = self.peer_pressure[student]
            print(f"{student} is experiencing {pressure} peer pressure.")

    def provide_guidance(self):
        for student in self.students.values():
            for teacher in self.teachers:
                print(f"{teacher} is providing guidance to {student}.")
                






class Student:
    def __init__(self, name, combat_skills, physical_health, weapons):
        self.name = name
        self.combat_skills = combat_skills
        self.physical_health = physical_health
        self.weapons = weapons
        
    def engage_in_combat(self, zombie):
        """Engages in combat with the given number of zombies"""
        # combat logic
        pass
    
    def update_health(self, health):
        """Updates the student's physical health"""
        self.physical_health = health
        
    def update_weapons(self, weapons):
        """Updates the student's weapons"""
        self.weapons = weapons
        

class School:
    def __init__(self, students, leaders, social_groups, alliances, hierarchy, supplies, weapons):
        self.students = students
        self.leaders = leaders
        self.social_groups = social_groups
        self.alliances = alliances
        self.hierarchy = hierarchy
        self.supplies = supplies
        self.weapons = weapons
        
    def form_alliance(self, student1, student2):
        """Forms an alliance between two students"""
        self.alliances.append((student1, student2))
        
    def assign_to_social_groups(self, student, group):
        """Assigns a student to a social group"""
        self.hierarchy[group].append(student)
        
    def enforce_rules_and_protocols(self, rules):
        """Enforces rules and protocols for the group"""
        # enforcing logic
        pass
    
    def scavenge_for_supplies(self, location):
        """Scavenges for supplies at a given location"""
        # scavenging logic
        pass
    
    def fortify_shelter(self, shelter):
        """Fortifies the group's shelter"""
        # fortifying logic
        pass
    
    def engage_in_combat(self, zombies):
        """Engages in combat with a given number of zombies"""
        # combat logic
        pass
    
    def update_supplies(self, supplies):
        """Updates the group's supplies"""
        self.supplies = supplies
        
    def update_weapons(self, weapons):
        """Updates the group's weapons"""
        self.weapons = weapons
