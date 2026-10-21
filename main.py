import cmd

from ctu_crs import CRS93, CRS97

ROBOTS = {"crs93": CRS93, "crs97": CRS97}


class RobotShell(cmd.Cmd):
    intro = "new intro"
    prompt = "crs> "

    def __init__(self):
        super().__init__()
        self.robot = None

    def emptyline(self):
        pass

    def do_init(self, arg):
        """init [crs93|crs97] - připojí se k robotu a provede hard a soft home (výchozí crs93)."""
        if self.robot is not None:
            print("Robot je už inicializovaný.")
            return
        name = arg.strip().lower() or "crs93"
        if name not in ROBOTS:
            print(f"Neznámý robot '{name}'. Možnosti: {', '.join(ROBOTS)}")
            return
        try:
            robot = ROBOTS[name]()
            robot.initialize()
        except Exception as e:
            print(f"Inicializace selhala: {e}")
            return
        self.robot = robot
        print(f"{name.upper()} inicializován.")

    def do_quit(self, arg):
        """quit - vrátí robota do soft home, zavře spojení a ukončí program."""
        if self.robot is not None:
            try:
                self.robot.soft_home()
                self.robot.close()
            except Exception as e:
                print(f"Chyba při ukončování: {e}")
            self.robot = None
        return True

    def do_EOF(self, arg):
        print()
        return self.do_quit(arg)


def main():
    shell = RobotShell()
    while True:
        try:
            shell.cmdloop()
            break
        except KeyboardInterrupt:
            # Ctrl+C jen zahodí řádek, nechceme se nechtěně odpojit od robota
            print("^C")
            shell.intro = None


if __name__ == "__main__":
    main()
