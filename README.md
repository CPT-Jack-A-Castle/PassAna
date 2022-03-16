# PassAna
更新日志：

22-03-16 优化部分代码结构，支持cpp python java JavaScript csharp

22-02-31 重构部分代码使得在添加新语言更容易
## 运行依赖
系统要求为Linux，
需要Codeql依赖。

## LGTM数据库准本
需要先将LGTM数据下载到本地，假设地址为 `$project-home$`

在`$project-home$`目录下使用 `for z in *.zip; do unzip $z; done` 直接解压所有zip文件，
注意只要保证解压出的文件不含有二级目录即可，也可用其他方法解压。

## 结构说明
`context`: 上下文分类器

`pwd`: password识别分类器

`ql`: 所有的分析QL语句和分析方法

`tokenizer`: NLP tokenizer模型


## 运行说明

`1.anaPass`: 查找所有的与password有关的string，输出为`pass.csv`

`1.anaStr`: 查找所有的string，输出为`string.csv`

`2.passContext`: 分析所有password有关的string的context, 包括前向和后向

`2.anaStr`: 分析所有string的context, 包括前向和后向

## 关于并行化
codeql并行运行会有程序锁，目前只有设置threads的方法，目前所有threads设置为8


