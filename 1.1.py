
def runge_kutta4(y_n, t_n, h, f):
    '''
    Gir y_n+1 ved hjelp av Runge Kutta 4.
    '''
    # Konstanter
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + (h/2)*k1)
    k3 = f(t_n + h/2, y_n + (h/2)*k2)
    k4 = f(t_n + h,   y_n + h * k3)
    
    # Regn ut og returner neste verdi
    return y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
hkkjj
