/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id python/example/empty-scope
 */

import python

from AssignStmt var, string txt, string namestr
where var.getValue() instanceof Str_ and txt = var.getValue().(StrConst).getText().toString() and
s.length() > 6 and s.length() < 256 and  s != "[empty string]" and
namestr = var.getATarget().toString().toLowerCase() and
	   contentstr = var.toString().toLowerCase() and
	(namestr.regexpMatch("\\w*password\\w*") or
     namestr.regexpMatch("\\w*passwd\\w*") or
      namestr.regexpMatch("\\w*pwd\\w*") or
      namestr.regexpMatch("\\w*secret\\w*") or
      namestr.regexpMatch("\\w*token\\w*") or
      namestr.regexpMatch("\\w*auth\\w*") or
      namestr.regexpMatch("\\w*host\\w*") or
      namestr.regexpMatch("\\w*server\\w*") or
      namestr.regexpMatch("\\w*ip\\w*") or
	  namestr.regexpMatch("\\w*username\\w*") or
      namestr.regexpMatch("\\w*account\\w*") or)
select var.getATarget(), var.getValue().(StrConst).getText().toString(), var.getLocation().getStartLine(), var.getLocation()