PROJECT_NAME := programming-interview

e:
	$(EDITOR) programming_interview.py
i:
	@python programming_interview.py -f 5
	@python programming_interview.py --fibonacci 7
	@python programming_interview.py --search
	@python programming_interview.py --sort
	@python programming_interview.py --queue
	@python programming_interview.py --list
	@python programming_interview.py --tree
	@python programming_interview.py --graph
