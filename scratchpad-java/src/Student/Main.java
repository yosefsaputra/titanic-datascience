package Student;

public class Main {
    public static void main(String[] args){
        Student a = new Student("yosef", 3, 29);
        Student b = new Student("novi", 4, 29);

        String[] strs = new String[]{"a", "b"};
        System.out.printf("%s and %s\n", strs);

        CompareStudent d = new CompareStudent();

        int c;
        c = d.compare(a, b);
        System.out.print(c);
    }
}
