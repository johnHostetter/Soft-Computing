"""
The following code was written by Seyed Saeid Masoumzadeh (GitHub user ID: seyedsaeidmasoumzadeh),
and was published for public use on GitHub under the MIT License at the following link:
    https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning 
"""

import copy
import numpy as np
import random
from empirical_fuzzy_set import EmpiricalFuzzySet as EFS
from environment import Environment
import matplotlib.pyplot as plt
import fuzzy_set as FuzzySet
import state_variable as StateVariable
import fuzzy_inference_system as FIS
import fuzzy_q_learning as FQL

test = [0, 0.0, 0.0034224598288770053, 0.0029944708894389916, 0.0038605503392438846, 0.0042956676007355026, 0.006890064517762412, 0.012510720511661277, 0.0, 0.0004278074224598931, 0.0004277846285793729, 0.0038537717610847336, 0.017578227892076893, 0.023108533463130403, 0.030421461682799605, 0.03547468246821121, 0.036397840361790425, 0.0679224845673543, -0.010694960912900228, -0.012877619340695822, -0.010893678015838857, -0.01015198877026708, -0.009410298210064696, -0.012822005735391655, -0.01374064213483985, -0.015482968908943484, -0.01599340580587747, -0.014869651755537998, -0.014968353684255111, -0.015882424888785773, -0.012726214058935616, -0.007135822969420054, -0.004804688700504444, -0.00041877883931900565, 0.0060272139412459675, 0.015791122792271434, 0.021364290154159936, 0.02523199493063206, 0.08817178973626755, 0.0, -0.00042780755080213897, -0.00042778491907712783, -0.0038466696134911585, -0.0059832739729124345, -0.006848251668888836, -0.005605121922821298, -0.0073065687201875045, -0.00479079058212554, -0.0035350175944774886, -0.0052355035824109145, -0.004818562090102663, -0.005245133747564976, -0.00862651564838189, -0.007790821717087162, -0.0069551525292046446, -0.009891730681497898, -0.011568395364707502, -0.009907714298694952, -0.010740613343184935, -0.011573459881941828, -0.016152873569016516, 0.0, 0.0029946523422459894, 0.0021388785817378847, 0.005592222690933435, 0.006895207040134849, 0.00949938366216668, 0.014714594194321265, 0.0, -0.008556014006554608, 0.0, -0.002566844983957219, -0.005133554113506479, -0.007700093834724572, -0.00899904330981445, -0.012396781176274034, -0.012865892588213744, -0.009185957601854294, -0.005091822912204265, -0.0034997550907007085, -0.0023279381588243147, -0.003683198935622174, -0.004616590321858162, -0.0026106345145889866, 0.0035797753193215175, 0.011449319307419258, 0.06548301329053478, 0.07490284139540779, 0.08192756162240505, 0.09040308421243234, 0.0, 0.0034224598288770053, 0.005561315809225087, 0.004249230046535229, 0.006839165072657207, 0.009428895014435229, 0.012453211982156872, 0.0, -0.0017112300106951871, -0.03835761488335756, -0.03843117233051568, -0.038889553081985474, -0.03819957874916022, -0.03675039583847569, -0.03417137294459893, -0.02822779304557595, -0.018569476423215478, -0.010020227719632279, -0.005156963442386625, 0.0011760833836464589, 0.006046416120532691, 0.01383113390094326, 0.023792395747682334, 0.0, -0.002994652470588235, -0.0017110715887559858, 0.0038193871930922007, 0.007645533949239207, 0.013629280784396607, 0.018741454499115538, 0.022086941654708814, 0.025431954161574493, 0.03192207394754659, 0.037052691206664067, 0.044017414681441984, 0.04772966309033144, 0.05144091955823115, 0.0522054848774529, 0.062464670882047026, 0.06710509173114182, 0.06606744359214514, 0.06778952100994202, 0.07236199165016467, 0.08720643253457508, 0.0, -0.025206977687776338, 0.0, -6.417112299465241e-11, -0.007272614111058787, 0.0, 0.004278074802139037, 0.010694960317675431, 0.014950001635572505, 0.020082072879210464, 0.04086644203391031, 0.044983684212590726, 0.0, 0.004278074802139037, 0.0, 0.0, -0.002994652470588235, -0.003850109021911066, -0.007253675598149134, -0.008537973177929864, -0.023732869878613245, -0.025941215550605544, -0.025723925151219892, 0.0, 0.0029946523422459894, 0.003422301041630933, 0.006866279266897592, 0.009447285530837374, 0.01289808636682039, 0.012846510246919487, 0.013677772982927173, 0.018037520693802725, 0.022396376231445317, 0.036497891000223784, 0.040775044829382164, 0.04598507116423313, 0.0, -0.00042780755080213897, 0.0034224824606020163, 0.006845217373922028, 0.011561740377134415, 0.021911856207684793, 0.0252992534849798, 0.023041309471437985, 0.021456044242108868]


# random play to collect data for empirical fuzzy sets
data = []
offline = {}
angle_list = []
reward = -1
env = Environment()
for iteration in range(0,200):
    if iteration % 200 == 0 or reward == -1:
        env.__init__()
    action = random.choice(env.action_set)
    temp = copy.deepcopy(env.state)
    temp.append(action)
    data.append(np.array(temp))
    reward, state_value = env.apply_action(action)
#    offline[iteration] = (state_value, action, reward)
    if reward != -1:
        angle_list.append(state_value[2])
        
plt.figure(figsize=(14,3))
plt.plot(angle_list)
plt.xlabel('Time')
plt.ylabel('Pole Angle with Random Action')
plt.show()

features = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip', 'Force']
efs = EFS(features)
class Data():
    def __init__(self, X, aen, episodes):
        self.X = X
        self.aen = aen
        self.episodes = episodes 
        
NFN_variables = efs.main(Data(data, None, None))



# random play to collect data for offline learning
data = []
offline = {}
angle_list = []
reward = -1
env = Environment()
for iteration in range(0,25000):
    if iteration % 200 == 0 or reward == -1:
        env.__init__()
    action = random.choice(env.action_set)
    temp = copy.deepcopy(env.state)
    temp.append(action)
    data.append(np.array(temp))
    reward, state_value = env.apply_action(action)
    offline[iteration] = (state_value, action, reward)
    if reward != -1:
        angle_list.append(state_value[2])
        
plt.figure(figsize=(14,3))
plt.plot(angle_list)
plt.xlabel('Time')
plt.ylabel('Pole Angle with Random Action')
plt.show()



# Create FIS
x1 = StateVariable.InputStateVariable()
x1.fuzzy_set_list = NFN_variables[0].terms
x2 = StateVariable.InputStateVariable()
x2.fuzzy_set_list = NFN_variables[1].terms
x3 = StateVariable.InputStateVariable()
x3.fuzzy_set_list = NFN_variables[2].terms
x4 = StateVariable.InputStateVariable()
x4.fuzzy_set_list = NFN_variables[3].terms

x1 = StateVariable.InputStateVariable(FuzzySet.Trapeziums(-2.4, -2, -1, -0.5), FuzzySet.Trapeziums(-1, -0.5, 0.5 , 1), FuzzySet.Trapeziums(0.5, 1, 2, 2.4) )
x2 = StateVariable.InputStateVariable(FuzzySet.Triangles(-2.4,-0.5,1), FuzzySet.Triangles(-0.5,1,2.4))
x3 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3.14159, -1.5, 0), FuzzySet.Triangles(-1.5, 0, 1.5), FuzzySet.Triangles(0, 1.5, 3.1459))
x4 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3.14159, -1.5, 0), FuzzySet.Triangles(-1.5, 0, 1.5), FuzzySet.Triangles(0, 1.5, 3.1459))
fis = FIS.Build(x1,x2,x3,x4)

# offline?

# Create Model
angle_list = []
model = FQL.Model(gamma = 0.9, alpha = 0.1 , ee_rate = 0.999, q_initial_value = 'zero',
                  action_set_length = 21, fis = fis)

print('Attempting to train offline...')
for _ in range(1):
    print('iteration..')
    for iteration in range(len(data)):
        (state, action, reward) = offline[iteration]
        model.run_offline(state, action, reward)
        

print(model.q_table)

# Create Model
angle_list = []
#model = FQL.Model(gamma = 0.9, alpha = 0.1 , ee_rate = 0.999, q_initial_value = 'random',
#                  action_set_length = 21, fis = fis)
env = Environment()
reward = -1
for iteration in range (0,1000):
    if iteration % 200 == 0 or reward == -1:
        if iteration % 200 == 0:
            print('controlled')
        else:
            print('failure')
        env.__init__()
        action = model.induce_policy(env.state, reward)
        reward, state_value = env.apply_action(action)
    action = model.induce_policy(state_value, reward)
    reward, state_value = env.apply_action(action)
    if reward != -1:
        angle_list.append(state_value[2])

plt.figure(figsize=(14,3))
plt.plot(angle_list)
plt.xlabel('Time')
plt.ylabel('Pole Angle')
plt.show()