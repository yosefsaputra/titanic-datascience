package Student;

import java.util.Comparator;

public class CompareStudent implements Comparator<Student> {
    @Override
    public int compare(Student a, Student b){
        if ((a.age < b.age) || ((a.age == b.age) && (a.name.compareTo(b.name) <= -1))){
            return -1;
        }
        else if (((a.age == b.age) && (a.name.compareTo(b.name) == 0))){
            return 0;
        }
        else{
            return 1;
        }
    }
}
