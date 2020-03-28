import sqlite3
import datetime


# basement
class QueryTemplate:
    def connect(self):
        self.conn = sqlite3.connect('sales.db')

    def construct_query(self):
        raise NotImplementedError()

    def do_query(self):
        results = self.conn.execute(self.query)
        self.results = results.fetchall()

    def format_results(self):
        output = []
        for row in self.results:
            row = [str(i) for i in row]
            output.append(', '.join(row))

        self.formatted_results = '\n'.join(output)

    def output_results(self):
        raise NotImplementedError()

    # 暴露给外部的方法
    def process_format(self):
        self.connect()
        self.construct_query()
        self.do_query()
        self.format_results()
        self.output_results()


class NewVehiclesQuery(QueryTemplate):
    def construct_query(self):
        self.query = "select * from Sales where new='true'"

    def output_results(self):
        print(self.formatted_results)


class UserGrossQuery(QueryTemplate):
    def construct_query(self):
        self.query = ("select salesperson, sum(amt) " +
                      " from Sales group by salesperson")

    def output_results(self):
        filename = "gross_sales_{0}".format(
            datetime.date.today().strftime('%Y%m%d')
        )

        with open(filename, 'w') as file:
            file.write(self.formatted_results)


def init():
    with sqlite3.connect("sales.db") as conn:
        try:
            conn.execute("CREATE TABLE Sales (salesperson text, "
                         "amt currency, year integer, model text, new boolean)")
            conn.execute("INSERT INTO Sales values"
                         " ('Tim', 16000, 2010, 'Honda Fit', 'true')")
            conn.execute("INSERT INTO Sales values"
                         " ('Tim', 9000, 2006, 'Ford Focus', 'false')")
            conn.execute("INSERT INTO Sales values"
                         " ('Gayle', 8000, 2004, 'Dodge Neon', 'false')")
            conn.execute("INSERT INTO Sales values"
                         " ('Gayle', 28000, 2009, 'Ford Mustang', 'true')")
            conn.execute("INSERT INTO Sales values"
                         " ('Gayle', 50000, 2010, 'Lincoln Navigator', 'true')")
            conn.execute("INSERT INTO Sales values"
                         " ('Don', 20000, 2008, 'Toyota Prius', 'false')")
            conn.commit()
        except sqlite3.OperationalError:
            print('Table Sales already exists')


if __name__ == '__main__':
    init()

    # 实质就是继承相同方法，重写不同之处来减少重复代码
    show = NewVehiclesQuery()
    show.process_format()

    save = UserGrossQuery()
    save.process_format()