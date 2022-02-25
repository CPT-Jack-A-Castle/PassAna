/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java

from Variable var, string namestr, string contentstr
where var.getType() instanceof  TypeString and
	   namestr = var.getName().toLowerCase() and
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
      namestr.regexpMatch("\\w*account\\w*") or
      contentstr.regexpMatch("\\w*com\\w*")
	)

select var, var.getInitializer(), var.getLocation().getStartLine(), var.getLocation()