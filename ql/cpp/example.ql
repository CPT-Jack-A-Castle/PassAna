/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id cpp/example/empty-block
 */

import cpp
import semmle.code.cpp.dataflow.TaintTracking
import DataFlow::PathGraph

predicate stepIn(Call c, DataFlow::Node arg, DataFlow::ParameterNode parm) {
    exists(int i | arg.asExpr() = c.getArgument(i) |
      parm.asParameter() = c.getTarget().getParameter(i))
  }
  
  predicate stepOut(Call c, DataFlow::Node ret, DataFlow::Node res) {
    exists(ReturnStmt retStmt | retStmt.getEnclosingFunction() = c.getTarget() |
      ret.asExpr() = retStmt.getExpr() and res.asExpr() = c)
  }
  
  predicate flowStep(DataFlow::Node pred, DataFlow::Node succ) {
    DataFlow::localFlowStep(pred, succ) or
    stepIn(_, pred, succ) or
    stepOut(_, pred, succ)
  }