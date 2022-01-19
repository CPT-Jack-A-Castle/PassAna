/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id cpp/example/empty-block
 */

import cpp


from Function arg
where
    arg.getName().regexpMatch("\\w*auth\\w*") or
    arg.getName().regexpMatch("\\w*remote\\w*") or
    arg.getName().regexpMatch("\\w*http\\w*") or
    arg.getName().regexpMatch("\\w*sock\\w*")

select arg, arg.getBasicBlock()
