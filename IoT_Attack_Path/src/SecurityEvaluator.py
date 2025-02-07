"""
CREST Project 9: Automated security assessment for interconnected systems
Created by Mengmeng Ge
Modified by Moyang Feng
05/02/2020
This module conducts security analysis and generates SHARPE code from HARM as text file.
"""

import sys
sys.path.append("..")
from IoT_Attack_Path.src.Harm import *
import math


#-------------------------------------------------------
#Compute CVSS Base Score
#-------------------------------------------------------

def computeBaseScore(harm):
    """
    Compute base score for HARM using attack graph as upper layer and attack tree as lower layer.
    """
    bs = []
    
    harm.model.calcBaseScore()
    counter = 1

    #print("=================================================")
    #print("#printing attack paths: \n")
    for path in harm.model.allpath:
        #print('[%d]' %counter, end = '\t')
        pathBS = 0
        for node in path:
            #print(node.name, end =' ')
            if node is not harm.model.s and node is not harm.model.e:
                if node.mv2.baseScore > 0:
                    #print('(%.2f)' %node.mv2.baseScore, end = ' ')
                    pathBS += node.mv2.baseScore
        #print('[Base Score = %.2f]\n' %pathBS)
        bs.append(pathBS)
        counter += 1
    # print("bs====",bs)
    indices = getMaxIndices(bs)
    val = bs[indices[0]]
    # print("indices====",indices)
    # print("indices[0]====",indices[0])
    # print("val====",val)
    #print("=================================================")
    #print("Maximum CVSS Base Score: %.2f\n" %val)
    path0 = printPaths_one(harm,indices)
    
    return val,indices,path0

def baseScoreAnalysis(net, pri):

    h = harm()
    h.constructHarm(net, "attackgraph", 1, "attacktree", 1, pri)
    #h.model.#printPath()
    #h.model.#printAG()
    
    if len(h.model.allpath) != 0:
        bs,indices,path0 = computeBaseScore(h)
        return bs,h,indices,path0
        
    return 0,h,0,[]

#-------------------------------------------------------
#Compute maximum risk
#-------------------------------------------------------
def computeRisk(harm):
    """
    Compute risk for HARM using attack graph as upper layer and attack tree as lower layer.
    """
    risk = []
    
    harm.model.calcRisk()
    counter = 1

    #print("=================================================")
    #print("#printing attack paths: \n")
    for path in harm.model.allpath:
        #print('[%d]' %counter, end = '\t')
        pathRisk = 0
        for node in path:
            #print(node.name, end =' ')
            if node is not harm.model.s and node is not harm.model.e:
                if node.mv2.risk > 0:
                    #print('(%.2f)' %node.mv2.risk, end = ' ')
                    pathRisk += node.mv2.risk
        #print('[Risk = %.2f]\n' %pathRisk)
        risk.append(pathRisk)
        counter += 1
    
    indices = getMaxIndices(risk)
    val = risk[indices[0]]

    #print("=================================================")
    #print("Maximum Risk: %.2f\n" %val)
    #printPaths(harm,indices)

    return val,indices

def riskAnalysis(net, pri):

    h = harm()
    h.constructHarm(net, "attackgraph", 1, "attacktree", 1, pri)
    #h.model.#printPath()
    #h.model.#printAG()

    if len(h.model.allpath) != 0:
        r,indices = computeRisk(h)
        return r,h,indices
    
    return 0,h,0

#-------------------------------------------------------
#Compute maximum attack impact
#-------------------------------------------------------

def computeAttackImpact(harm):
    """
    Compute attack impact for HARM using attack graph as upper layer and attack tree as lower layer.
    """
    impact = []
    
    harm.model.calcImpact()
    counter = 1

    #print("=================================================")
    #print("#printing attack paths: \n")
    for path in harm.model.allpath:
        #print('[%d]' %counter, end = '\t')
        pathImpact = 0
        for node in path:
            #print(node.name, end =' ')
            if node is not harm.model.s and node is not harm.model.e:
                if node.mv2.impactScore > 0:
                    #print('(%.2f)' %node.mv2.impactScore, end = ' ')
                    pathImpact += node.mv2.impactScore
        #print('[Impact = %.2f]\n' %pathImpact)
        impact.append(pathImpact)
        counter += 1
    
    indices = getMaxIndices(impact)
    val = impact[indices[0]]
    
    #print("=================================================")
    #print("Maximum attack impact score: %.2f\n" %val)
    #printPaths(harm,indices)
    
    return val,indices

def attackImpactAnalysis(net, pri):

    h = harm()
    h.constructHarm(net, "attackgraph", 1, "attacktree", 1, pri)
    #h.model.#printPath()
    #h.model.#printAG()
    
    if len(h.model.allpath) != 0:
        ai,indices = computeAttackImpact(h)
        return ai,h,indices
    
    return 0,h,0

#---------------------------------------------------------------
#Compute maximum attack success probability 
#---------------------------------------------------------------

def computeAttackPro(harm):
    """
    Compute attack success probability for HARM using attack graph as upper layer and attack tree as lower layer.
    """
    pro = []

    harm.model.calcPro()
    counter = 1
    
    #print("=================================================")
    #print("#printing attack paths: \n")
    for path in harm.model.allpath:
        #print('[%d]' %counter, end = '\t')
        pathPro = 1
        for node in path:
            if node is not harm.model.s and node is not harm.model.e:
                #print(node.name, end =' ')
                #Exclude the attacker
                if node.mv2.probability > 0:
                    #print('(%.2f)' %node.mv2.probability, end = ' ')
                    pathPro *= node.mv2.probability
        #print('[Pro = %.2f]\n' %pathPro)
        pro.append(pathPro)
        counter += 1
    
    indices = getMaxIndices(pro)
    val = pro[indices[0]]

    #print("=================================================")
    #print('Maximum attack success probability: %.2f\n' %val)
    #printPaths(harm,indices)
    return val,indices

def attackProAnalysis(net, pri):

    h = harm()
    h.constructHarm(net, "attackgraph", 1, "attacktree", 1, pri)
    #h.model.#printPath()
    #h.model.#printAG()
    
    if len(h.model.allpath) != 0:
        pro,indices = computeAttackPro(h)
        return pro,h,indices
        
    return 0,h,0

#------------------------------------
#Compute the number of paths
#------------------------------------
def NP_metric(harm):
    
    value = len(harm.model.allpath)
    return value

#------------------------------------------
#Compute the mean of path lengths
#------------------------------------------
def MPL_metric(harm):

    sum_path_length = 0
    for path in harm.model.allpath:
        sum_path_length += int(len(path)-3)
        ##print(sum_path_length)

    value = float(sum_path_length/len(harm.model.allpath))

    return value

#----------------------------------------
#Compute the mode of path lengths
#----------------------------------------
def MoPL_metric(harm):
    
    NP = []
    for path in harm.model.allpath:
        NP.append(int(len(path)-3))

    value = max(NP, key=NP.count)
    return value

#----------------------------------------------------------
#Compute the standard deviation of path lengths
#----------------------------------------------------------
def SDPL_metric(harm):

    sumation_DPL = 0
    MPL = MPL_metric(harm)
    ##print(MPL)
    for path in harm.model.allpath:    
        sumation_DPL += float(len(path) - 3 - MPL)**2
        ##print(sumation_DPL)

    value = math.sqrt(float(sumation_DPL / len(harm.model.allpath)))

    return value

#--------------------------------------
#Compute the shortest attack path
#--------------------------------------
def SP_metric(harm):

    SP=[]
    for path in harm.model.allpath:
        SP.append(int(len(path)-3))
    value = min(SP)
    return value

# -------------------------------------
# #print attack paths
# -------------------------------------
def printPaths_one(harm,indices):
    #print('Found %d attack paths: ' % len(indices))
    for i in indices:
        path = harm.model.allpath[i]
        path0 = []

        for node in path:
            if node is not harm.model.s and node is not harm.model.e:
                path0.append(node.name[3:])

        return path0
                # #print(node.name, end =' -> ')
        #print('END\n')
    return []
def printPaths_all(harm,indices):
    #print('Found %d attack paths: ' % len(indices))
    attackpath = []
    for i in indices:
        path = harm.model.allpath[i]
        path0 = []
        #print('[%d]' %(i+1), end = '\t')
        for node in path:
            if node is not harm.model.s and node is not harm.model.e:
                path0.append(node.name[3:])
        attackpath.append(path0)
                # #print(node.name, end =' -> ')
        #print('END\n')
    return attackpath
# -------------------------------------
# Get the indices with the maximum value
# -------------------------------------
def getMaxIndices(vals):
    maximum = 0.0
    indices = []
    for i in range(len(vals)): 
        if vals[i] > maximum:
            maximum = vals[i]
            indices.clear()
            indices.append(i)
        elif vals[i] == maximum:
            indices.append(i)

    return indices
