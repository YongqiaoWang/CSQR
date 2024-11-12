# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log, NonNegativeReals, Reals
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd

from .constant import CET_ADDI, CET_MULT, FUN_PROD, FUN_COST, RTS_CRS, RTS_VRS, OPT_LOCAL, OPT_DEFAULT
from .utils import tools, interpolation


class LSQR:
    """Linear superquantile regression (LSQR)
    """

    def __init__(self, y, x, tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """LSQR model

        Args:
            y (float): output variable. 
            x (float): input variables.
            tau (float): quantile.
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x, self.z = tools.assert_valid_basic_data(y, x, z)
        self.tau = tau
        self.cet = cet
        self.fun = fun
        self.rts = rts

        self.nSample = len(self.y)
        self.nInput = len(self.x[0])
        self.alpha = 0
        self.beta = np.zeros(self.nInput)
        # self.epsilon = np.zeros(self.nSample)
        
        self.kStart = np.ceil(self.nSample*self.tau).astype('int')-1
        self.t = (1+np.arange(self.nSample))/self.nSample
        self.t[self.kStart-1] = self.tau
        self.paraz = np.zeros(self.nSample)
        self.paraE = np.zeros(self.nSample)
        for k in range(self.kStart,self.nSample-1):
            self.paraz[k] = self.t[k]-self.t[k-1]
            self.paraE[k] = np.log(1-self.t[k-1])-np.log(1-self.t[k])
        self.paraz = self.paraz/(1-self.tau)
        self.paraE = self.paraE/(self.nSample*(1-self.tau))
       
        # Initialize the LSQR model
        self.__model__=ConcreteModel('Step 1 of linear superquantile regression')

        if type(self.z) != type(None):
            # Initialize the set of z
            self.__model__.K = Set(initialize=range(len(self.z[0])))

            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.K, doc='z coefficient')
            
        # Initialize the sets
        self.__model__.I = Set(initialize = range(self.nSample))
        self.__model__.J = Set(initialize = range(self.nInput))
        self.__model__.IK = Set(initialize = range(self.kStart, self.nSample - 1))

        # Initialize the variables
        # self.__model__.alpha = Var(within=Reals, doc='alpha')
        self.__model__.beta = Var(self.__model__.J, within=Reals, doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, within=Reals, doc='epsion')
        self.__model__.z = Var(self.__model__.IK, within=Reals, doc='z aux')
        self.__model__.E = Var(self.__model__.IK, self.__model__.I, domain=NonNegativeReals, doc='E')
        self.__model__.w = Var(within=Reals, doc='w')

        self.__model__.epsilon_plus = Var(self.__model__.I, doc='positive error term')
        self.__model__.epsilon_minus = Var(self.__model__.I, doc='negative error term')
        self.__model__.frontier = Var(self.__model__.I,
                                        bounds=(0.0, None),
                                        doc='estimated frontier')
        
        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')


        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        # if self.cet == CET_MULT:
        #     self.__model__.log_rule = Constraint(self.__model__.I,
        #                                          rule=self.__log_rule(),
        #                                          doc='log-transformed regression equation')
       
        self.__model__.E_rule = Constraint(self.__model__.IK,
                                           self.__model__.I, 
                                           rule=self.__E_rule(),
                                           doc='E rule')  

        self.__model__.w_rule = Constraint(self.__model__.I,
                                         rule=self.__w_rule(),
                                         doc='w rule')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        print('The first step of superquantile regression............')
        # TODO(error/warning handling): Check problem status after optimization
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, self.cet, solver)

        if (self.optimization_status == 0):
            print('The first step of linear superquantile regression failed............')
            return
        
        tools.assert_optimized(self.optimization_status)

        self.epsilon = self.get_residual()

        #the second step of convex superquantile regression
        self.__model2__=ConcreteModel('Step 2 of linear superquantile regression')

        # Initialize the sets
        self.__model2__.I = Set(initialize=range(self.nSample))

        # Initialize the variables
        self.__model2__.e = Var(self.__model2__.I, domain=NonNegativeReals, doc='e')
        self.__model2__.a0 = Var(doc = 'a0')

        # Setup the objective function and constraints
        self.__model2__.objective = Objective(rule=self.__objective2_rule(),
                                             sense=minimize,
                                             doc='objective function')

        self.__model2__.e_rule = Constraint(self.__model2__.I,
                                            rule=self.__e_rule(),
                                            doc='e rule')

        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model2__, email, self.cet, solver)

        if (self.optimization_status == 0):
            print('The second step of linear superquantile regression failed............')
            return
        
        #self.__model__.alpha.set_value(self.__model2__.objective())
        self.alpha=self.__model2__.objective()
        for j in self.__model__.J:
            self.beta[j]=self.__model__.beta[j].value
        
        for i in self.__model__.I:
            # self.__model__.epsilon[i].set_value(self.y[i]-self.__model2__.objective()-np.dot(self.beta,self.x[i]))
            self.__model__.epsilon[i].set_value(self.__model__.epsilon[i].value - self.__model2__.objective())
            self.__model__.epsilon_plus[i].set_value(np.max(self.__model__.epsilon[i].value,0))
            self.__model__.epsilon_minus[i].set_value(np.max(-self.__model__.epsilon[i].value, 0))


        print("Two steps of convex superquantile regression is done.....")
        
        tools.assert_optimized(self.optimization_status)

    def __objective_rule(self):
        """Return the proper objective function"""
        
        def objective_rule(model):
            return sum(model.z[k]*self.paraz[k] for k in self.__model__.IK)\
                    +model.w/(self.nSample*(1-self.tau))\
                    +sum(self.paraE[k]*sum(model.E[k,i] for i in self.__model__.I) for k in self.__model__.IK)\
                    -sum(model.epsilon[i] for i in self.__model__.I)/self.nSample

        return objective_rule

    # def __error_decomposition(self):
    #     """Return the constraint decomposing error to positive and negative terms"""
    #
    #     def error_decompose_rule(model, i):
    #         return model.epsilon[i] == model.epsilon_plus[i] - model.epsilon_minus[i]
    #
    #     return error_decompose_rule
    
    def __E_rule(self):
        """Return the constraint for E"""

        def E_rule(model,k,i):
            return model.epsilon[i]-model.z[k]<=model.E[k,i]
            
        return E_rule

    def __w_rule(self):
        """Return the constraint for w"""
        
        def w_rule(model,i):
            return model.epsilon[i]<=model.w
        
        return w_rule
    
    def __regression_rule(self):
        """Return the proper regression constraint"""
        
        # def regression_rule(model, i):
        #     return self.y[i] == np.dot(self.x[i],model.beta) + model.epsilon[i]

        # return regression_rule
        if self.cet == CET_ADDI:
            if self.rts == RTS_VRS:
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return self.y[i] == np.dot(self.x[i],model.beta)\
                            + sum(model.lamda[k] * self.z[i][k]
                                  for k in model.K) + model.epsilon[i]

                    return regression_rule

                def regression_rule(model, i):
                    return self.y[i] == np.dot(self.x[i],model.beta) \
                        + model.epsilon[i]

                return regression_rule
            elif self.rts == RTS_CRS:
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return self.y[i] == np.dot(self.x[i],model.beta) \
                            + sum(model.lamda[k] * self.z[i][k]
                                  for k in model.K) + model.epsilon[i]

                    return regression_rule

                def regression_rule(model, i):
                    return self.y[i] == np.dot(self.x[i],model.beta) \
                        + model.epsilon[i]

                return regression_rule

        elif self.cet == CET_MULT:
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return log(self.y[i]) == log(model.frontier[i] + 1) \
                        + sum(model.lamda[k] * self.z[i][k]
                              for k in model.K) + model.epsilon[i]

                return regression_rule

            def regression_rule(model, i):
                return log(self.y[i]) == log(model.frontier[i] + 1) + model.epsilon[i]

            return regression_rule

        raise ValueError("Undefined model parameters.")

 
    def __objective2_rule(self):
        """Return the proper objective function"""

        def objective2_rule(model):
            return model.a0+sum(model.e[i] for i in self.__model2__.I)/(self.nSample * (1 - self.tau))

        return objective2_rule

    def __e_rule(self):
        """Return the constraint for w"""

        def e_rule(model, i):
            return model.e[i] >= self.epsilon[i]-model.a0

        return e_rule

    def display_status(self):
        """Display the status of problem"""
        print(self.optimization_status)

    def display_alpha(self):
        """Display alpha value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_various_return_to_scale(self.rts)
        self.__model__.alpha.display()

    def display_beta(self):
        """Display beta value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.beta.display()

    def display_lamda(self):
        """Display lamda value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_contextual_variable(self.z)
        self.__model__.lamda.display()

    def display_residual(self):
        """Dispaly residual value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.epsilon.display()

    def display_positive_residual(self):
        """Dispaly positive residual value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.epsilon_plus.display()

    def display_negative_residual(self):
        """Dispaly negative residual value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.epsilon_minus.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_alpha(self):
        """Return alpha value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_various_return_to_scale(self.rts)
        alpha = self.__model2__.a0.value
        #alpha = list(np.array(list(self.__model__.alpha[:].value)) + self.__model2__.a0.value)
        return np.asarray(alpha)

    def get_beta(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        beta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta),
                                                          list(self.__model__.beta[:, :].value))])
        beta = pd.DataFrame(beta, columns=['Name', 'Key', 'Value'])
        beta = beta.pivot(index='Name', columns='Key', values='Value')
        return beta.to_numpy()

    def get_lamda(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_contextual_variable(self.z)
        lamda = list(self.__model__.lamda[:].value)
        return np.asarray(lamda)

    def get_residual(self):
        """Return residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual = list(self.__model__.epsilon[:].value)
        return np.asarray(residual)

    def get_positive_residual(self):
        """Return positive residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual_plus = list(self.__model__.epsilon_plus[:].value)
        return np.asarray(residual_plus)

    def get_negative_residual(self):
        """Return negative residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual_minus = list(self.__model__.epsilon_minus[:].value)
        return np.asarray(residual_minus)

    def get_frontier(self):
        """Return estimated frontier value by array"""
        tools.assert_optimized(self.optimization_status)
        if self.cet == CET_MULT:
            frontier = np.asarray(list(self.__model__.frontier[:].value)) + 1
        elif self.cet == CET_ADDI:
            frontier = np.asarray(self.y) - self.get_residual()
        return np.asarray(frontier)
    
    def get_predict(self, x_test):
        """Return the estimated function in testing sample"""
        tools.assert_optimized(self.optimization_status)
        return self.alpha*np.ones(np.size(x_test,0))+np.dot(x_test,self.beta)
